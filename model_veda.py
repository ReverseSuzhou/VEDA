# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py
from typing import Tuple
from collections import OrderedDict
import math
import functools
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import split_first_dim_linear
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from vit_new import *
import torch.nn.init as init
from clip_adapter import clip
from thop import profile
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear



CLIP_VIT_B16_PATH = './checkpoint_backbone/ViT-B-16.pt'
CLIP_VIT_L14_PATH = ''
DWCONV3D_DISABLE_CUDNN = True
NUM_SAMPLES = 1
np.random.seed(83)
torch.manual_seed(83)
torch.cuda.manual_seed(83)
torch.cuda.manual_seed_all(83)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(dim, down_dim)
        self.D_fc2 = nn.Linear(down_dim, dim)

        nn.init.constant_(self.D_fc2.weight, 0.)
        nn.init.constant_(self.D_fc2.bias, 0.)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class S_Adapter_attn(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)  # 降维1/4
        kernel_size = (3, 3)
        self.fc1 = nn.Linear(dim, down_dim)
        self.down_dim = down_dim
        self.conv = nn.Conv2d(
            down_dim, down_dim,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=down_dim,
        )
        self.fc2 = nn.Linear(down_dim, dim)
        self.gelu = nn.GELU()
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc2.weight, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
        
    def forward(self, x): # (197,40,dim)
        T = 8
        x_id = x
        x = self.fc1(x) 
        x_res = x
        x = x.permute(1, 0, 2)  #(40,197,dim/4)

        q = x
        k = x.permute(0, 2, 1)
        v = x

        scores = torch.matmul(q, k) / math.sqrt(self.down_dim)
        attn_weights = F.softmax(scores, dim=-1)
        x = torch.matmul(attn_weights, v)

        x = x.permute(1, 0, 2)
        # x = self.gelu(x)
        x = x + x_res
        x = self.fc2(x)
        if self.skip_connect:
            x_id = x_id + x
        else:
            x_id = x
        
        return x_id

class S_Adapter_eca(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)
        kernel_size = (3, 3)
        self.fc1 = nn.Linear(dim, down_dim)
        self.down_dim = down_dim
        k_size = 3
        self.conv1d = nn.Conv1d(down_dim, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(down_dim, dim)
        self.gelu = nn.GELU()

        nn.init.constant_(self.fc2.weight, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
        
    def forward(self, x):
        T = 8
        x_id = x
        x = self.fc1(x) 

        avg_pool = torch.mean(x, dim=1, keepdim=True)

        attention = self.conv1d(avg_pool.transpose(-1, -2))
        attention = self.sigmoid(attention)

        x = x * attention

        x = self.fc2(x)
        if self.skip_connect:
            x_id = x_id + x
        else:
            x_id = x
        
        return x_id

class T_Adapter_attn(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)
        self.down_dim = down_dim
        kernel_size = (3, 3)
        self.fc1 = nn.Linear(dim, down_dim)
        self.conv = nn.Conv1d(in_channels=down_dim, out_channels=down_dim, kernel_size=3, padding=1, groups=down_dim)

        self.fc2 = nn.Linear(down_dim, dim)
        self.gelu = nn.GELU()
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc2.weight, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
        
    def forward(self, x):
        T = 8
        x_id = x
        x = self.fc1(x) 
        x_res = x
        x = x.permute(1, 0, 2)
        q = x
        k = x.permute(0, 2, 1)
        v = x

        scores = torch.matmul(q, k) / math.sqrt(self.down_dim)
        attn_weights = F.softmax(scores, dim=-1)
        x = torch.matmul(attn_weights, v)
            
  
        x = x.permute(1, 0, 2)
        x = x + x_res
        x = self.fc2(x)
        if self.skip_connect:
            x_id = x_id + x
        else:
            x_id = x
        
        return x_id

class Dense_Adapter_gai(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, skip_connect = True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)
        kernel_size = (3, 1, 1)
        self.fc1 = nn.Linear(dim, down_dim)
        self.conv = nn.Conv3d(
            down_dim, down_dim,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=down_dim,
        )
        self.fc2 = nn.Linear(down_dim, dim)
        self.gelu = nn.GELU()
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc2.weight, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = 8
        x_id = x
        x = self.fc1(x)  
        x = x.permute(1, 0, 2)
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x = x[:, 1:, :]  # (40,196,dim)
        x_cls = x[:, 0, :].unsqueeze(1)  # (40,1,dim)
        # x = self.fc1(x)  # (40,196,dim/4)
        x_temp = x.view(B, T, L - 1, Ca)   # (5,8,196,dim/4)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous() # (5,8,14,14,dim/4)->(5,dim/4,8,14,14)

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)  # (5,dim/4,8,14,14)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca) # (5,dim/4,8,14,14)->(40,196,dim/4)
        x_after = x.view(B, T, L - 1, Ca)   # (5,8,196,dim/4)

        '''针对x_temp和x_after进行操作'''
        x_temp_split = x_temp.unsqueeze(2)
        x_after_split = x_after.unsqueeze(1)


        device = x.device
        diff = x_after_split - x_temp_split
        mask = torch.triu(torch.ones(8, 8), diagonal=1).to(device)
        mask = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


        result = (diff * mask).sum(dim=(1, 2))


        result = result / 28

        expanded_result = result.unsqueeze(1).expand(-1, 8, -1, -1).contiguous()
        x_last = x_temp + expanded_result
        x_last = x_last.view(BT, -1, Ca)#(40,197,dim/4)
        x = torch.cat((x_cls, x_last), dim=1) 
        
        x = x.permute(1, 0, 2)

        x = self.gelu(x)
        x = self.fc2(x)
        if self.skip_connect:
            x_id = x_id + x
        else:
            x_id= x

        return x_id

class Dense_Adapter_eca(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, skip_connect = True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)
        kernel_size = (3, 1, 1)
        self.fc1 = nn.Linear(dim, down_dim)
        self.conv = nn.Conv3d(
            down_dim, down_dim,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=down_dim,
        )
        self.fc2 = nn.Linear(down_dim, dim)
        self.gelu = nn.GELU()
        k_size = 3
        self.conv1d = nn.Conv1d(down_dim, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc2.weight, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        T = 8
        x_id = x
        x = self.fc1(x)  
        x = x.permute(1, 0, 2)# (40,196,dim/4)
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x = x[:, 1:, :]  # (40,196,dim)
        x_cls = x[:, 0, :].unsqueeze(1)  # (40,1,dim)
        # x = self.fc1(x)  # (40,196,dim/4)
        x_temp = x.view(B, T, L - 1, Ca)   # (5,8,196,dim/4)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous() # (5,8,14,14,dim/4)->(5,dim/4,8,14,14)

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        x = self.conv(x)  # (5,dim/4,8,14,14)
        torch.backends.cudnn.enabled = cudnn_enabled

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca) # (5,dim/4,8,14,14)->(40,196,dim/4)
        x_after = x.view(B, T, L - 1, Ca)   # (5,8,196,dim/4)


        x_temp_split = x_temp.unsqueeze(2)  # 将 x_temp 扩展为 (5, 8, 1, 196, dim)
        x_after_split = x_after.unsqueeze(1)  # 将 x_after 扩展为 (5, 1, 8, 196, dim)

        # 计算差值并处理上三角矩阵
        device = x.device
        diff = x_after_split - x_temp_split  # 形状为 (5, 8, 8, 196, dim)
        mask = torch.triu(torch.ones(8, 8), diagonal=1).to(device)  # 上三角矩阵（不含对角线），形状为 (8, 8)
        mask = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # 扩展为 (1, 8, 8, 1, 1) 用于广播

        # 应用 mask 并求和
        result = (diff * mask).sum(dim=(1, 2))


        result = result / 28

        expanded_result = result.unsqueeze(1).expand(-1, 8, -1, -1).contiguous().view(BT, -1, Ca) # (40,196,dim/4)
        
        avg_pool = torch.mean(expanded_result, dim=1, keeepdim=True) # (40,1,dim/4)
        attention = self.conv1d(avg_pool.transpose(-1, -2))  # (batch, 1, dim) -> (batch, 1, dim)
        attention = self.sigmoid(attention)  # (batch, 1, dim)
        x_last = expanded_result * attention          

        x = torch.cat((x_cls, x_last), dim=1) 
        
        x = x.permute(1, 0, 2)

        x = self.gelu(x)
        x = self.fc2(x)
        if self.skip_connect:
            x_id = x_id + x
        else:
            x_id= x

        return x_id


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.num_frames = 8
        self.drop_path = 0.
        self.scale = 0.5

        self.T_Adapter = T_Adapter_attn(d_model, skip_connect=False)
        self.S_Adapter = S_Adapter_eca(d_model, skip_connect=True)
        self.S_MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.drop_path = DropPath(drop_path) if self.drop_path > 0. else nn.Identity()
        self.ST_Adapter = Dense_Adapter_gai(d_model, skip_connect=False)



    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, s_x: torch.Tensor):  # x.shape = (197,40,768)   
        n, bt, d = s_x.shape
        s_x_temp = s_x
        # 时间分支
        xt = rearrange(s_x, 'n (b t) d -> t (b n) d', t=self.num_frames)  # (8,985,768)
        xt = self.T_Adapter(self.attention(self.ln_1(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)   # (197,40,768)
        t_x = self.drop_path(xt) # skip connection original + time attention result

        #
        s_x = s_x_temp + self.S_Adapter(self.attention(self.ln_1(s_x_temp))) # original space multi head self attention  

        s_x = s_x + t_x
        s_x = s_x + self.ST_Adapter(s_x)

        s_xn = self.ln_2(s_x)
        s_x = s_x + self.mlp(s_xn) + self.drop_path(self.scale * self.S_MLP_Adapter(s_xn))  
 
        return s_x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        for n, p in self.named_parameters():
            if 'Adapter'not in n:
                p.requires_grad_(False)
                p.data = p.data

    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, 3, 224, 224)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
       
        x = self.ln_post(x[:, 0, :])  # (B*T,1,768)
        if self.proj is not None:
            x = x @ self.proj
        # x = self.ln_post(x)  # (B*T,197,768)
        return x


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class vlr_vis(nn.Module):

    def __init__(self, num_seq, dim):
        super().__init__()
        self.t = num_seq
        self.dim = dim
        self.decoder_layers = 4
        vit = vit_base()
        self.decoder_blocks = vit.clip_blocks[-self.decoder_layers:]

        width = dim
        scale = width ** -0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # self.positional_embedding = nn.Parameter(scale * torch.randn(self.t + 1, width))
        self.norm = vit.norm

        N = 100
        self.object_queries = nn.Parameter(torch.rand(N, self.dim))
         
        self.linear = nn.Linear(N, self.t)

        self.norm_final = nn.LayerNorm(dim)

    def forward(self, x):  # 这里输入x是encoder的输出，也就是（B*T,dim)
        x = x.view(-1, self.t, self.dim) # (B,T,dim)
        B = x.shape[0]
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [b, t+ 1, width]
        # x = x + self.positional_embedding.to(x.dtype)
        object_query = self.object_queries.unsqueeze(0).repeat(B, 1, 1) # (B,N,dim) 

        for blk in self.decoder_blocks:
            object_query = blk(object_query, x)

        object_query = object_query.permute(0, 2, 1)
        object_query = self.linear(object_query)
        object_query = object_query.permute(0, 2, 1).contiguous().view(-1,self.dim) # (5,8,dim)  ## 这里也tm写错了 (40,768)


        split_queries = torch.unbind(object_query, dim=1)

        orthogonal_queries = []

        for i in range(0, len(split_queries), 2):
            vec1 = split_queries[i]
            vec2 = split_queries[i + 1]
            
            vec1_normalized = F.normalize(vec1, p=2, dim=-1)
            vec2_projected = vec2 - torch.sum(vec2 * vec1_normalized, dim=-1, keepdim=True) * vec1_normalized
            vec2_normalized = F.normalize(vec2_projected, p=2, dim=-1)
            
            orthogonal_queries.append(vec1_normalized)
            orthogonal_queries.append(vec2_normalized)

        orthogonal_object_query = torch.stack(orthogonal_queries, dim=1)
        object_query = self.norm_final(orthogonal_object_query)


        return object_query

class vlr_text(nn.Module):
    def __init__(self, num_seq, dim):
        super().__init__()
        self.t = num_seq
        self.dim = dim
        self.decoder_layers = 4
        vit = vit_base()
        self.decoder_blocks = vit.clip_blocks[-self.decoder_layers:]

        width = dim
        scale = width ** -0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # self.positional_embedding = nn.Parameter(scale * torch.randn(self.t + 1, width))
        self.norm = vit.norm

        N = 100
        self.object_queries = nn.Parameter(torch.rand(N, self.dim))
         
        self.linear = nn.Linear(N, self.t)

        self.norm_final = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.view(-1, self.t, self.dim) # (B,T,dim)
        B = x.shape[0]
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [b, t+ 1, width]
        # x = x + self.positional_embedding.to(x.dtype)
        object_query = self.object_queries.unsqueeze(0).repeat(B, 1, 1) # (B,N,dim) 

        for blk in self.decoder_blocks:
            object_query = blk(object_query, x)

        object_query = object_query.permute(0, 2, 1)
        object_query = self.linear(object_query)
        object_query = object_query.permute(0, 2, 1).contiguous().view(-1,self.dim) # (5,8,dim)  ## 这里也tm写错了 (40,768))


        return object_query



class text_image_cross(nn.Module):   
    def __init__(self, num_seq, dim):
        super().__init__()
        self.t = num_seq
        self.dim = dim
        self.decoder_layers = 4
        vit = vit_base()
        self.decoder_blocks = vit.self_attn_blocks[-self.decoder_layers:]

        width = dim
        scale = width ** -0.5

        self.norm = vit.norm
        self.object_queries = nn.Parameter(torch.empty(self.dim))
        init.normal_(self.object_queries, mean=0.0, std=1.0)
        self.norm_final = nn.LayerNorm(self.dim)

    def vec(self, object_query):
        split_queries = torch.unbind(object_query, dim=1)

        orthogonal_queries = []

        for i in range(0, len(split_queries), 2):
            vec1 = split_queries[i]
            vec2 = split_queries[i + 1]
            
            vec1_normalized = F.normalize(vec1, p=2, dim=-1)
            vec2_projected = vec2 - torch.sum(vec2 * vec1_normalized, dim=-1, keepdim=True) * vec1_normalized
            vec2_normalized = F.normalize(vec2_projected, p=2, dim=-1)

            orthogonal_queries.append(vec1_normalized)
            orthogonal_queries.append(vec2_normalized)

        orthogonal_object_query = torch.stack(orthogonal_queries, dim=1)
        object_query = self.norm_final(orthogonal_object_query)

    def forward(self, text_features, x):
        x = x.view(-1, self.t, self.dim) # (B,T,dim)
        B = x.shape[0]

        object_query = self.object_queries.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) # (B,1,dim) 
        text_features = text_features.view(-1, 1, self.dim)  # (B, 1, dim)

        fusion = torch.cat((object_query, text_features, x), dim=1)  # (B,2+self.t,dim)

        for blk in self.decoder_blocks:
            fusion = blk(fusion)
        object_query = fusion[:, 0, :]
        object_query = object_query.view(-1,self.dim) #  (40,768)
        # object_query = self.vec(object_query)

        return object_query  # 这个作为最后的特征

class vlf(nn.Module):
    def __init__(self, num_seq, dim):
        super().__init__()
        self.t = num_seq
        self.dim = dim
        self.decoder_layers = 6
        vit = vit_base()
        self.decoder_blocks = vit.clip_blocks[-self.decoder_layers:]

        width = dim
        scale = width ** -0.5

        self.norm = vit.norm
        self.object_queries = nn.Parameter(torch.empty(self.dim))
        init.normal_(self.object_queries, mean=0.0, std=1.0)
        self.norm_final = nn.LayerNorm(self.dim)

    def vec(self, object_query):

        split_queries = torch.unbind(object_query, dim=1)

        orthogonal_queries = []

        for i in range(0, len(split_queries), 2):
            vec1 = split_queries[i]
            vec2 = split_queries[i + 1]
            
            vec1_normalized = F.normalize(vec1, p=2, dim=-1)
            vec2_projected = vec2 - torch.sum(vec2 * vec1_normalized, dim=-1, keepdim=True) * vec1_normalized
            vec2_normalized = F.normalize(vec2_projected, p=2, dim=-1)
            
            orthogonal_queries.append(vec1_normalized)
            orthogonal_queries.append(vec2_normalized)

        orthogonal_object_query = torch.stack(orthogonal_queries, dim=1)
        object_query = self.norm_final(orthogonal_object_query)

    def forward(self, text_features, x):  # x是motion
        text_features = text_features.view(-1, self.t, self.dim) # (B,T,dim)
        x = x.view(-1, self.t, self.dim) # (B,T,dim)
        B = x.shape[0]

        for blk in self.decoder_blocks:
            text_features = blk(text_features, x)

        text_features = text_features.view(-1, self.dim)

        return text_features

class FSAR_CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.visual = VisionTransformer(
            input_resolution=args.input_resolution,
            output_dim=args.output_dim,
            patch_size=args.patch_size,
            width=args.width,
            layers=args.layers,
            heads=args.heads,
        )

        assert CLIP_VIT_B16_PATH is not None, 'Please set CLIP_VIT_B16_PATH in configs.py'
        checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
        print(self.visual.load_state_dict(checkpoint.visual.state_dict(), strict=False))

        parameters = list(checkpoint.state_dict().keys())
        
        self.vlr_vis = vlr_vis(num_seq=self.args.seq_len, dim=args.output_dim)
        self.vlr_text = vlr_text(num_seq=self.args.seq_len, dim=args.output_dim)
        self.vlf = vlf(num_seq=self.args.seq_len, dim=args.output_dim)

        self.class_real_train = args.CLASS_NAME_TRAIN                                                                                                                   
        self.class_real_test = args.CLASS_NAME_TEST
        self.clip_model, preprogress = clip.load("ViT-B/16")


        self.alpha_image = nn.Parameter(torch.tensor(0.5))
        self.alpha_motion = nn.Parameter(torch.tensor(1.0))
        self.alpha_fusion_image = nn.Parameter(torch.tensor(0.5))
        self.alpha_fusion_motion = nn.Parameter(torch.tensor(0.5))


        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.softmax3 = nn.Softmax(dim=-1)
        self.softmax4 = nn.Softmax(dim=-1)


    def calculate_euclidean_distance(self, support, query):
        support = support.unsqueeze(0)  # shape becomes (1, 25, 768)
        query = query.unsqueeze(1)  # shape becomes (20, 1, 768)
        distance = torch.sqrt(((support - query) ** 2).sum(-1))  # shape becomes (20, 25)
        return distance

    def cos_sim(self, x, y, epsilon=0.01):
        """
        Calculates the cosine similarity between the last dimension of two tensors.
        """
        numerator = torch.matmul(x, y.transpose(-1, -2))
        xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
        ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
        denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
        dists = torch.div(numerator, denominator)
        return dists

    def extract_class_indices(self, labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
    
    def get_text_feats (self, real_support_labels):
        train_text_template = ["A Video of Human is Doing {}".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
        train_text_template.sort()
        test_text_template = ["A Video of Human is Doing {}".format(self.class_real_test[int(ii)]) for ii in range(len(self.class_real_test))]
        test_text_template.sort()

        train_token = clip.tokenize(train_text_template).to(device)
        test_token = clip.tokenize(test_text_template).to(device)
        

        text_train = self.clip_model.encode_text(train_token)
        text_test = self.clip_model.encode_text(test_token)


        if self.training:
            support_text_features = text_train[real_support_labels]
        else:
            support_text_features = text_test[real_support_labels]

        support_text_features = support_text_features.float()
        return support_text_features
    
    def get_freeze_clip_image(self, x):
        x = self.clip_model.encode_image(x)
        return x

    def get_query_text_feats(self, n_queries):
        text_template = "A Video of Human is Doing Action Something"

        token = clip.tokenize(text_template).to(device)
        with torch.no_grad():
            single_feature = self.clip_model.encode_text(token).float()  # shape: (1, dim)
        query_text_features = single_feature.repeat(n_queries, 1)  # shape: (n_queries, dim)

        query_text_features = query_text_features.float()
        return query_text_features
    
    def get_bimhd_logits(self, support_features, query_features, n_support, n_queries, context_labels):
        unique_labels = torch.unique(context_labels)
        frame_sim = self.cos_sim(query_features, support_features)
        frame_dists = 1 - frame_sim
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)

        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)
        class_dists = [
            torch.mean(torch.index_select(cum_dists, 1, self.extract_class_indices(context_labels, c)), dim=1) for
            c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        frame_logist = - class_dists
        return frame_logist

    def forward(self, context_images, context_labels, target_images, real_support_labels):
        n_support = context_images.shape[0]
        n_queries = target_images.shape[0]

        support_text_features = self.get_text_feats(real_support_labels).view(-1, 1, self.args.output_dim).repeat(1, self.args.seq_len, 1)  # (B, 8, dim)
        query_text_features = self.get_query_text_feats(n_queries).view(-1, 1, self.args.output_dim).repeat(1, self.args.seq_len, 1)  # (B, 8, dim)
        support_features = self.visual(context_images)  # (bs_s*t, 197,768)
        query_features = self.visual(target_images)    # (bs_q*t, 197, 768)


        support_features_motion = self.vlr_vis(support_features)
        query_features_motion = self.vlr_vis(query_features)
        support_text_features_motion = self.vlr_text(support_text_features)
        query_text_features_motion = self.vlr_text(query_text_features)

        support_features_fusion_image = self.vlf(support_text_features, support_features)
        query_features_fusion_image = self.vlf(query_text_features, query_features)
        support_features_fusion_motion = self.vlf(support_text_features_motion, support_features_motion)
        query_features_fusion_motion = self.vlf(query_text_features_motion, query_features_motion)

        unique_labels = torch.unique(context_labels)
        logits_image = self.get_bimhd_logits(support_features=support_features, query_features=query_features, n_support=n_support, n_queries=n_queries, context_labels=context_labels)
        logits_motion = self.get_bimhd_logits(support_features=support_features_motion, query_features=query_features_motion, n_support=n_support, n_queries=n_queries, context_labels=context_labels)
        logits_fusion_image = self.get_bimhd_logits(support_features=support_features_fusion_image, query_features=query_features_fusion_image, n_support=n_support, n_queries=n_queries, context_labels=context_labels)
        logits_fusion_motion = self.get_bimhd_logits(support_features=support_features_fusion_motion, query_features=query_features_fusion_motion, n_support=n_support, n_queries=n_queries, context_labels=context_labels)



        frame_logits = self.alpha_image * logits_image + self.alpha_motion * logits_motion + self.alpha_fusion_image * logits_fusion_image + self.alpha_fusion_motion * logits_fusion_motion

        return_dict = {'logits': split_first_dim_linear(frame_logits, [NUM_SAMPLES, n_queries])}
        return return_dict




    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.visual.cuda(0)
            self.visual = torch.nn.DataParallel(self.visual, device_ids=[i for i in range(0, self.args.num_gpus)])




if __name__ == '__main__':
    class ArgsObject():
        def __init__(self):
            self.way = 5
            self.shot = 1
            self.seq_len = 8
            self.image_size = 224
            self.query_per_class = 2
            self.input_resolution=224
            self.patch_size=16
            self.width=768
            self.output_dim=512
            self.layers=2
            self.heads=12
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 512

            self.CLASS_NAME_TRAIN = ['Pouring [something] into [something]', 'Poking a stack of [something] without the stack collapsing', 'Pretending to poke [something]', 'Lifting up one end of [something] without letting it drop down', 'Moving [part] of [something]', 'Moving [something] and [something] away from each other', 'Removing [something], revealing [something] behind', 'Plugging [something] into [something]', 'Tipping [something] with [something in it] over, so [something in it] falls out', 'Stacking [number of] [something]', "Putting [something] onto a slanted surface but it doesn't glide down", 'Moving [something] across a surface until it falls down', 'Throwing [something] in the air and catching it', 'Putting [something that cannot actually stand upright] upright on the table, so it falls on its side', 'Holding [something] next to [something]', 'Pretending to put [something] underneath [something]', "Poking [something] so lightly that it doesn't or almost doesn't move", 'Approaching [something] with your camera', 'Poking [something] so that it spins around', 'Pushing [something] so that it falls off the table', 'Spilling [something] next to [something]', 'Pretending or trying and failing to twist [something]', 'Pulling two ends of [something] so that it separates into two pieces', 'Lifting up one end of [something], then letting it drop down', "Tilting [something] with [something] on it slightly so it doesn't fall down", 'Spreading [something] onto [something]', 'Touching (without moving) [part] of [something]', 'Turning the camera left while filming [something]', 'Pushing [something] so that it slightly moves', 'Uncovering [something]', 'Moving [something] across a surface without it falling down', 'Putting [something] behind [something]', 'Attaching [something] to [something]', 'Pulling [something] onto [something]', 'Burying [something] in [something]', 'Putting [number of] [something] onto [something]', 'Letting [something] roll along a flat surface', 'Bending [something] until it breaks', 'Showing [something] behind [something]', 'Pretending to open [something] without actually opening it', 'Pretending to put [something] onto [something]', 'Moving away from [something] with your camera', 'Wiping [something] off of [something]', 'Pretending to spread air onto [something]', 'Holding [something] over [something]', 'Pretending or failing to wipe [something] off of [something]', 'Pretending to put [something] on a surface', 'Moving [something] and [something] so they collide with each other', 'Pretending to turn [something] upside down', 'Showing [something] to the camera', 'Dropping [something] onto [something]', "Pushing [something] so that it almost falls off but doesn't", 'Piling [something] up', 'Taking [one of many similar things on the table]', 'Putting [something] in front of [something]', 'Laying [something] on the table on its side, not upright', 'Lifting a surface with [something] on it until it starts sliding down', 'Poking [something] so it slightly moves', 'Putting [something] into [something]', 'Pulling [something] from right to left', 'Showing that [something] is empty', 'Spilling [something] behind [something]', 'Letting [something] roll down a slanted surface', 'Holding [something] behind [something]']
            self.CLASS_NAME_TEST = ['Twisting (wringing) [something] wet until water comes out', 'Poking a hole into [something soft]', 'Pretending to take [something] from [somewhere]', 'Putting [something] upright on the table', 'Poking a hole into [some substance]', 'Rolling [something] on a flat surface', 'Poking a stack of [something] so the stack collapses', 'Twisting [something]', '[Something] falling like a feather or paper', 'Putting [something] on the edge of [something] so it is not supported and falls down', 'Pushing [something] off of [something]', 'Dropping [something] into [something]', 'Letting [something] roll up a slanted surface, so it rolls back down', 'Pushing [something] with [something]', 'Opening [something]', 'Putting [something] on a surface', 'Taking [something] out of [something]', 'Spinning [something] that quickly stops spinning', 'Unfolding [something]', 'Moving [something] towards the camera', 'Putting [something] next to [something]', 'Scooping [something] up with [something]', 'Squeezing [something]', 'Failing to put [something] into [something] because [something] does not fit']
            self.num_gpus = 1

    args = ArgsObject()
    support_images = torch.rand(args.way, args.seq_len, 3, args.image_size, args.image_size).to(device)
    support_labels = torch.tensor([0, 1, 2, 3, 4]).to(device)
    real_support_labels = torch.tensor([11, 12, 31, 5, 55]).to(device)
    query_images = torch.rand(args.way * args.query_per_class, args.seq_len, 3, args.image_size, args.image_size).to(device)
    model = FSAR_CNN(args).to(device)
    if args.num_gpus > 1:
         model.distribute_model()
    

    out = model(support_images, support_labels, query_images, real_support_labels)

    print(out)
    print(out)