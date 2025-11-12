import torch
from torchvision import transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
from transformations.augmentation import KineticsResizedCropFewshot, ColorJitter, Compose
from transformations.random_erasing import RandomErasing
import torchvision.transforms._transforms_video as transforms
from numpy.random import randint

"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []

    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)

        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)


"""Dataset for few-shot videos, which returns few-shot tasks. """
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.get_item_counter = 0

        self.data_dir = args.path  # 数据集地址
        self.seq_len = args.seq_len  # 8，一个视频几个帧表示
        self.train = True

        self.img_size = args.img_size
        self.annotation_path = args.traintestlist  # 数据集few-shot划分地址

        self.way = args.way
        self.shot = args.shot
        self.query_per_class = args.query_per_class

        self.train_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self._select_fold()
        self.read_dir()

    def setup_transforms(self):
        # transform_train
        std_transform_list_query = [
            transforms.ToTensorVideo(),
            transforms.RandomHorizontalFlipVideo(),
            KineticsResizedCropFewshot(
                short_side_range=[256, 256],
                crop_size=224,
            ), ]
        std_transform_list = [
            transforms.ToTensorVideo(),
            KineticsResizedCropFewshot(
                short_side_range=[256, 256],
                crop_size=224,
            ),
            # transforms.RandomHorizontalFlipVideo()
        ]
        std_transform_list_query.append(
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.25,
                grayscale=0.3,
                consistent=True,
                shuffle=True,
                gray_first=True,
                is_split=False
            ),
        )
        std_transform_list_query += [
            transforms.NormalizeVideo(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
                inplace=True
            ),
            RandomErasing()
        ]
        std_transform_list += [
            transforms.NormalizeVideo(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
                inplace=True
            ),
        ]
        # transform_train

        # transform_test
        resize_video = KineticsResizedCropFewshot(
            short_side_range=[256, 256],  # [256, 256]
            crop_size=224,  # 224
            num_spatial_crops=1,  # 1
            idx=True
        )  # KineticsResizedCrop
        std_transform_list_test = [
            transforms.ToTensorVideo(),
            resize_video,
            transforms.NormalizeVideo(
                mean=[0.485, 0.456, 0.406],  # [0.45, 0.45, 0.45]
                std=[0.229, 0.224, 0.225],  # [0.225, 0.225, 0.225]
                inplace=True
            )
        ]

        self.transform = {}
        self.transform["train_support"] = Compose(std_transform_list)
        self.transform["train_query"] = Compose(std_transform_list_query)
        self.transform["test"] = Compose(std_transform_list_test)

    """Loads all videos into RAM from an uncompressed zip. Necessary as the filesystem has a large block size, which is unsuitable for lots of images. """
    """Contains some legacy code for loading images directly, but this has not been used/tested for a while so might not work with the current codebase. """
    def read_dir(self):
        # load zipfile into memory
        if self.data_dir.endswith('.zip'):
            self.zip = True
            zip_fn = os.path.join(self.data_dir)
            # 因为压缩包压缩的就是二进制文件
            self.mem = open(zip_fn, 'rb').read()    # 压缩包以2进制文件打开，
            self.zfile = zipfile.ZipFile(io.BytesIO(self.mem))  # BytesIO开辟IO缓存，多用于存储图片视频信息。StringIO存储字符信息
        else:       # self.zfile包含所有图片的压缩信息             # 不过和文件一样，使用完记得及时关闭回收内存空间。
            self.zip = False

        # go through zip and populate splits with frame locations and action groundtruths
        if self.zip:
            # ssv2数据集中的图片格式原本就是jpg，提取出不包含.jpg的文件路径，也即一些目录文件之类的，因此称之为dir_list，剩下的就是图片文件
            # When using 'png' based datasets like kinetics, replace 'jpg' to 'png'
            dir_list = list(set([x for x in self.zfile.namelist() if '.jpg' not in x]))
            # x.split("/")) > 2目的是提取那些文件格式文件夹如下的：“ssv2/push/2048/”，差分后["ssv2","push","2048"," "],-3就是类别，所以觉class_folders
            # 但是有一个类是ssv2压缩包的名字，也要注意
            class_folders = list(set([x.split("/")[-3] for x in dir_list if len(x.split("/")) > 2]))
            class_folders.sort()
            self.class_folders = class_folders   # video_folders同上，-2是2048就是值得是视频的文件夹，ssv2理论上有1万个视频文件
            video_folders = list(set([x.split("/")[-2] for x in dir_list if len(x.split("/")) > 3]))
            video_folders.sort()
            self.video_folders = video_folders
            # 重新指定索引标号：这样每个视频文件夹和所有类别(甚至包含数据集压缩包名字)，压缩包名字小写，所以索引号100（第101个类别）
            class_folders_indexes = {v: k for k, v in enumerate(self.class_folders)}
            video_folders_indexes = {v: k for k, v in enumerate(self.video_folders)}
            # 该操作挑选出压缩包内所有.jpg文件地址
            img_list = [x for x in self.zfile.namelist() if '.jpg' in x]
            img_list.sort()  # 排列方式就按照了和class_folders的一样排列方式（因为图片文件夹路径，类别靠前）
            # video_folders[0] = “100001”号视频的编号， c默认其实就是的train_split,不过后面也没用到这值
            c = self.get_train_or_test_db(video_folders[0])

            last_video_folder = None
            last_video_class = -1
            insert_frames = []
            for img_path in img_list:
                # [-3:0]，其值分别表示为类别文件夹，视频文件夹和视频每帧的jpg文件夹。
                class_folder, video_folder, jpg = img_path.split("/")[-3:]

                if video_folder != last_video_folder:
                    # 忽略小于seq.len的视频
                    if len(insert_frames) >= self.seq_len:
                        c = self.get_train_or_test_db(last_video_folder.lower())
                        if c != None:
                            c.add_vid(insert_frames, last_video_class)
                        else:
                            pass
                    insert_frames = []
                    class_id = class_folders_indexes[class_folder]
                    vid_id = video_folders_indexes[video_folder]

                insert_frames.append(img_path)
                last_video_folder = video_folder
                last_video_class = class_id

            c = self.get_train_or_test_db(last_video_folder)
            if c != None and len(insert_frames) >= self.seq_len:
                c.add_vid(insert_frames, last_video_class)
        else:
            # 非zip文件处理方式，直接得出所有文件夹
            class_folders = os.listdir(self.data_dir)
            class_folders.sort()
            self.class_folders = class_folders
            for class_folder in class_folders:
                # 得到该类所有视频文件夹目录
                video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
                video_folders.sort()
                # self.args.debug_loader作用是Load 1 vid per class for debugging
                if self.args.debug_loader:
                    video_folders = video_folders[0:1]
                for video_folder in video_folders:
                    # 返回视频属于测试还是训练的划分集
                    c = self.get_train_or_test_db(video_folder)
                    if c == None:
                        continue
                    # 根据目前类别视频文件夹得到所有该视频文件夹下的所有视频帧为imgs
                    imgs = os.listdir(os.path.join(self.data_dir, class_folder, video_folder))
                    # 忽略小于seq.len的视频
                    if len(imgs) < self.seq_len:
                        continue
                    imgs.sort()  # 这里path也是列表，所以一个视频占split类的一个值
                    paths = [os.path.join(self.data_dir, class_folder, video_folder, img) for img in imgs]
                    paths.sort()
                    class_id = class_folders.index(class_folder)
                    c.add_vid(paths, class_id)
        print("loaded {}".format(self.data_dir))
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))

    """ return the current split being used ，就是返回测试还是训练的划分集"""
    def get_train_or_test_db(self, split=None):
        if split is None:  # is None表示有值，说明读取的压缩包有视频文件夹
            get_train_split = self.train
        else:
            if split in self.train_test_lists["train"]:
                get_train_split = True
            elif split in self.train_test_lists["test"]:
                get_train_split = False
            else:
                return None
        if get_train_split:
            return self.train_split
        else:
            return self.test_split

    """ load the paths of all videos in the train and test splits. """
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name, self.args.split)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                # train list种data的长度为6400，每类100个视频，然后就是训练集64类，所以是64个视频，list中元素类别为str
                data = fid.readlines()
                # str.replace(old, new，[, max])，第一个元素是被替换元素，第二个是替换元素，第三个是替换的最大次数
                # 通过此把类别中的空格换成了"_"
                data = [x.replace(' ', '_').lower() for x in data]
                # strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列,注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
                # 对应的结果就是读取的每行的视频信息最后的“\n”消失，split(" ")消除空格
                data = [x.strip().split(" ")[0] for x in data]
                # os.path.splitext(“文件路径”) ,分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作,本例其实没有扩展名
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                # os.path.split(x)[1]表示取后面的视频指标数字
#                 if "kinetics" in self.args.path:
#                     data = [x[0:11] for x in data]
                # extend表示往列表一次添加多个元素
                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """

    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        return len(c)

    """ Get the classes used for the current split """

    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes

    """Loads a single image from a specified path """

    def read_single_image(self, path):
        if self.zip:
            with self.zfile.open(path, 'r') as f:
                with Image.open(f) as i:
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i.load()
                return i

    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        num_segments = 8
        seg_length = 1
        total_length = num_segments * seg_length

        n_frames = len(paths)
        if n_frames == self.args.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            num_frames = n_frames
            offsets = list()
            ticks = [i * num_frames // num_segments for i in range(num_segments + 1)]
            for i in range(num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= seg_length:
                    tick += randint(tick_len - seg_length + 1)
                offsets.extend([j for j in range(tick, tick + seg_length)])
            idxs = offsets

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train_support"]
            else:
                transform = self.transform["test"]
            imgs = [torch.tensor(np.array(v)) for v in imgs]
            imgs = torch.stack(imgs)
            imgs = transform(imgs)
            imgs = imgs.transpose(1, 0)
        return imgs, vid_id

    def get_seq_query(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        n_frames = len(paths)
        num_segments = 8
        seg_length = 1
        total_length = num_segments * seg_length
        num_frames = n_frames
        if n_frames == self.args.seq_len:
            idxs = [int(f) for f in range(n_frames)]

        else:
            offset = (num_frames / num_segments - seg_length) / 2.0
            out = np.array([i * num_frames / num_segments + offset + j
                            for i in range(num_segments)
                            for j in range(seg_length)], dtype=np.int)
            idxs = [int(f) for f in out]

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train_query"]
            else:
                transform = self.transform["test"]
            imgs = [torch.tensor(np.array(v)) for v in imgs]
            imgs = torch.stack(imgs)

            imgs = transform(imgs)
            imgs = imgs.transpose(1, 0)
        return imgs, vid_id, idxs

    """returns dict of support and target images and labels"""

    def __getitem__(self, index):
        # select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []
        ids_list = []
        for bl, bc in enumerate(batch_classes):
            # select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)

            for idx in idxs[0:self.args.shot]:
                vid, vid_id = self.get_seq(bc, idx)
                # vid, vid_id, ids = self.get_seq_query(bc, idx)
                support_set.append(vid)
                support_labels.append(bl)
                real_support_labels.append(bc)

            for idx in idxs[self.args.shot:]:
                vid, vid_id, ids = self.get_seq_query(bc, idx)
                ids_list.append(ids)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)
        s = list(zip(support_set, support_labels, real_support_labels))
        random.shuffle(s)
        support_set, support_labels, real_support_labels = zip(*s)

        t = list(zip(target_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)

        support_set = torch.cat(support_set)

        target_set = torch.cat(target_set)

        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        real_support_labels = torch.FloatTensor(real_support_labels)
        batch_classes = torch.FloatTensor(batch_classes)
        return {"support_set": support_set, "support_labels": support_labels, "target_set": target_set,
                "target_labels": target_labels, "real_support_labels": real_support_labels,
                "real_target_labels": real_target_labels, "batch_class_list": batch_classes}


