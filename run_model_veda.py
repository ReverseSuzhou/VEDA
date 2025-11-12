import torch
import numpy as np
import argparse
import os
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy
from model_veda import FSAR_CNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf
import math
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import video_reader_test
import random
import logging
import json
from thop import profile
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# logger for training accuracies
train_logger = setup_logger('Training_accuracy', './run_model/train_output.log')

# logger for evaluation accuracies
eval_logger = setup_logger('Evaluation_accuracy', './run_model/eval_output.log')

#############################################
# setting up seeds
manualSeed = 18
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)


########################################################

def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        json_path = './class_indices_train_hmdb.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            self.class_indict = json.load(f)

        json_path = './class_indices_test_hmdb.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            self.class_indict_test = json.load(f)
        
        args = self.parse_command_line()
        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = 'cuda:0'
        self.device = torch.device(gpu_device)
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        self.vd = video_reader_test.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)

        self.loss = loss
        self.accuracy_fn = aggregate_accuracy

        params_with_decay, params_without_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if '.bias' in n:
                params_without_decay.append(p)
            else:
                params_with_decay.append(p)
        self.optimizer = torch.optim.Adam(
            [
                {'params': params_with_decay, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
                {'params': params_without_decay, 'lr': args.learning_rate, 'weight_decay': 0.}
            ],
        )

        self.test_accuracies = TestAccuracies(self.test_set)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_func)

        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        self.args.input_resolution=224
        self.args.patch_size=16
        self.args.width=768
        self.args.layers=12
        self.args.heads=12
        self.args.output_dim=512
        self.args.temp_set = [2, 3]
        self.args.trans_linear_in_dim = 512
        self.args.trans_linear_out_dim = 512
        self.args.trans_dropout = 0.1
        model = FSAR_CNN(
            args=self.args
        )
        model = model.to(self.device)
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set

    def lr_func(self, step):  
        epoch = step / self.args.epochs_per_class
        if epoch < self.args.warmup_epochs:
            return epoch / self.args.warmup_epochs
        else:
            return 0.5 + 0.5 * math.cos((epoch - self.args.warmup_epochs) / (self.args.epochs - self.args.warmup_epochs) * math.pi)

    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf", "diving"], default="hmdb",
                            help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.00005, help="Learning rate.")
        parser.add_argument('--weight_decay', type=float, default=5e-5,
                            help='optimizer weight decay.')
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m",
                            default='/home',
                            help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=100020,
                            help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False,
                            action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=2,
                            help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=2,
                            help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', nargs='+', type=int,
                             help='iterations to test at. Default is for ssv2 otam split.',
                             default=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000,10500,11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500,
                                      16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000,
                            26000, 28000, 30000, 32000,34000, 36000,38000, 40000,42000, 44000,46000, 48000,50000,52000,54000,
                             56000,58000, 60000, 62000, 64000, 66000, 68000, 70000, 72000, 74000, 76000, 78000, 80000, 82000, 84000,
                            86000, 88000, 90000, 92000, 94000, 96000, 98000, 100000, 102000,104000, 106000, 108000,110000, 112000,114000, 116000,118000, 120000,122000, 124000,
                             126000, 128000, 130000, 132000,134000, 136000,138000, 140000, 142000, 144000, 146000, 148000,150000,152000,154000,
                             156000,158000, 160000, 162000, 164000, 166000, 168000, 170000, 172000, 174000, 176000, 178000, 180000, 182000, 184000,
                             186000, 188000, 190000, 192000, 194000, 196000, 198000, 200000])

        parser.add_argument("--num_test_tasks", type=int, default=1000, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=50, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=24, help="Num dataloader workers.")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="adam", help="Optimizer")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--save_freq", type=int, default=500,
                            help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--scratch", choices=["bc", "bp", "new"], default="bp",
                            help="directory containing dataset, splits, and checkpoint saves.")
        parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true",
                            help="Load 1 vid per class for debugging")
        parser.add_argument("--split", type=int, default=3, help="Dataset split.")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument("--test_model_only", type=bool, default=False,
                            help="Only testing the model from the given checkpoint")
        parser.add_argument('--num_frames', type=int, default=8, dest='num_frames',
                            help='disable fp16 during training or inference')
        parser.add_argument('--warmup_epochs', type=int, default=1,
                            help='number of warmup epochs.')
        parser.add_argument('--epochs_per_class', type=int, default=10000,
                            help='num tasks one epochs.')
        parser.add_argument('--epochs', type=int, default=5,
                            help='num tasks one epochs.')
        parser.add_argument("--emb_q_num", type=int, default=4, help="可学习压缩帧数的个数")



        args = parser.parse_args()
        # hmdb
        args.CLASS_NAME_TRAIN = ['brush hair', 'catch', 'chew', 'clap', 'climb', 'climb stairs', 'dive', 'draw_sword', 'dribble', 'drink', 'fall floor', 'flic flac', 'handstand', 'hug', 'jump', 'kiss', 'pullup', 'punch', 'push', 'ride_bike', 'ride horse', 'shake_hands', 'shoot_bow', 'situp', 'stand', 'sword', 'sword exercise', 'throw', 'turn', 'walk', 'wave']
        args.CLASS_NAME_TEST = ['fencing', 'kick', 'kick ball', 'pick', 'pour', 'pushup', 'run', 'sit', 'smoke', 'talk']

        # # ucf
        # args.CLASS_NAME_TRAIN = ['Apply Eye Makeup', 'Archery', 'Baby Crawling', 'Balance Beam', 'Band Marching', 'Baseball Pitch', 'Basketball', 'Basketball Dunk', 'Bench Press', 'Biking', 'Billiards', 'Blow DryHair', 'Body Weight Squats', 'Bowling', 'Boxing Punching Bag', 'Boxing Speed Bag', 'Breast Stroke', 'Brushing Teeth', 'Cricket Bowling', 'Drumming', 'Fencing', 'Field Hockey Penalty', 'Frisbee Catch', 'Front Crawl', 'Haircut', 'Hammering', 'Head Massage', 'Hula Hoop', 'Javelin Throw', 'Juggling Balls', 'Jumping Jack', 'Kayaking', 'Knitting', 'Long Jump', 'Lunges', 'Military Parade', 'Mixing', 'Mopping Floor', 'Nunchucks', 'Parallel Bars', 'Pizza Tossing', 'Playing Cello', 'Playing Dhol', 'Playing Flute', 'Playing Piano', 'Playing Sitar', 'Playing Tabla', 'Playing Violin', 'Pole Vault', 'Pull Ups', 'Push Ups', 'Rafting', 'Rope Climbing', 'Rowing', 'Shaving Beard', 'Skijet', 'Soccer Juggling', 'Soccer Penalty', 'Sumo Wrestling', 'Swing', 'Table Tennis Shot', 'Tai Chi', 'Throw Discus', 'Trampoline Jumping', 'Typing', 'Uneven Bars', 'Walking WithDog', 'Wall Pushups', 'Writing On Board', 'Yo Yo']
        # args.CLASS_NAME_TEST = ['Blowing Candles', 'Clean And Jerk', 'Cliff Diving', 'Cutting in Kitchen', 'Diving', 'Floor Gymnastics', 'Golf Swing', 'Handstand Walking', 'Horse Race', 'Ice Dancing', 'Jump Rope', 'Pommel Horse', 'Punch', 'Rock Climbing Indoor', 'Salsa Spin', 'Skiing', 'Sky Diving', 'Still Rings', 'Surfing', 'Tennis Swing', 'Volleyball Spiking']

        # # kinetics
        # args.CLASS_NAME_TRAIN =  ['air drumming', 'arm wrestling', 'beatboxing', 'biking through snow', 'blowing glass', 'blowing out candles', 'bowling', 'breakdancing', 'bungee jumping', 'catching or throwing baseball', 'cheerleading', 'cleaning floor', 'contact juggling', 'cooking chicken', 'country line dancing', 'curling hair', 'deadlifting', 'doing nails', 'dribbling basketball', 'driving tractor', 'drop kicking', 'dying hair', 'eating burger', 'feeding birds', 'giving or receiving award', 'hopscotch', 'jetskiing', 'jumping into pool', 'laughing', 'making snowman', 'massaging back', 'mowing lawn', 'opening bottle', 'playing accordion', 'playing badminton', 'playing basketball', 'playing didgeridoo', 'playing ice hockey', 'playing keyboard', 'playing ukulele', 'playing xylophone', 'presenting weather forecast', 'punching bag', 'pushing cart', 'reading book', 'riding unicycle', 'shaking head', 'sharpening pencil', 'shaving head', 'shot put', 'shuffling cards', 'slacklining', 'sled dog racing', 'snowboarding', 'somersaulting', 'squat', 'surfing crowd', 'trapezing', 'using computer', 'washing dishes', 'washing hands', 'water skiing', 'waxing legs', 'weaving basket']
        # args.CLASS_NAME_TEST = ['blasting sand',  'busking',  'cutting watermelon',  'dancing ballet', 'dancing charleston',  'dancing macarena',  'diving cliff', 'filling eyebrows', 'folding paper',  'hula hooping', 'hurling (sport)',  'ice skating',  'paragliding', 'playing drums',  'playing monopoly', 'playing trumpet', 'pushing car', 'riding elephant',  'shearing sheep', 'side kick', 'stretching arm', 'tap dancing', 'throwing axe',  'unboxing']

        # # ssv2_small
        # args.CLASS_NAME_TRAIN = ['Pouring [something] into [something]', 'Poking a stack of [something] without the stack collapsing', 'Pretending to poke [something]', 'Lifting up one end of [something] without letting it drop down', 'Moving [part] of [something]', 'Moving [something] and [something] away from each other', 'Removing [something], revealing [something] behind', 'Plugging [something] into [something]', 'Tipping [something] with [something in it] over, so [something in it] falls out', 'Stacking [number of] [something]', "Putting [something] onto a slanted surface but it doesn't glide down", 'Moving [something] across a surface until it falls down', 'Throwing [something] in the air and catching it', 'Putting [something that cannot actually stand upright] upright on the table, so it falls on its side', 'Holding [something] next to [something]', 'Pretending to put [something] underneath [something]', "Poking [something] so lightly that it doesn't or almost doesn't move", 'Approaching [something] with your camera', 'Poking [something] so that it spins around', 'Pushing [something] so that it falls off the table', 'Spilling [something] next to [something]', 'Pretending or trying and failing to twist [something]', 'Pulling two ends of [something] so that it separates into two pieces', 'Lifting up one end of [something], then letting it drop down', "Tilting [something] with [something] on it slightly so it doesn't fall down", 'Spreading [something] onto [something]', 'Touching (without moving) [part] of [something]', 'Turning the camera left while filming [something]', 'Pushing [something] so that it slightly moves', 'Uncovering [something]', 'Moving [something] across a surface without it falling down', 'Putting [something] behind [something]', 'Attaching [something] to [something]', 'Pulling [something] onto [something]', 'Burying [something] in [something]', 'Putting [number of] [something] onto [something]', 'Letting [something] roll along a flat surface', 'Bending [something] until it breaks', 'Showing [something] behind [something]', 'Pretending to open [something] without actually opening it', 'Pretending to put [something] onto [something]', 'Moving away from [something] with your camera', 'Wiping [something] off of [something]', 'Pretending to spread air onto [something]', 'Holding [something] over [something]', 'Pretending or failing to wipe [something] off of [something]', 'Pretending to put [something] on a surface', 'Moving [something] and [something] so they collide with each other', 'Pretending to turn [something] upside down', 'Showing [something] to the camera', 'Dropping [something] onto [something]', "Pushing [something] so that it almost falls off but doesn't", 'Piling [something] up', 'Taking [one of many similar things on the table]', 'Putting [something] in front of [something]', 'Laying [something] on the table on its side, not upright', 'Lifting a surface with [something] on it until it starts sliding down', 'Poking [something] so it slightly moves', 'Putting [something] into [something]', 'Pulling [something] from right to left', 'Showing that [something] is empty', 'Spilling [something] behind [something]', 'Letting [something] roll down a slanted surface', 'Holding [something] behind [something]']
        # args.CLASS_NAME_TEST = ['Twisting (wringing) [something] wet until water comes out', 'Poking a hole into [something soft]', 'Pretending to take [something] from [somewhere]', 'Putting [something] upright on the table', 'Poking a hole into [some substance]', 'Rolling [something] on a flat surface', 'Poking a stack of [something] so the stack collapses', 'Twisting [something]', '[Something] falling like a feather or paper', 'Putting [something] on the edge of [something] so it is not supported and falls down', 'Pushing [something] off of [something]', 'Dropping [something] into [something]', 'Letting [something] roll up a slanted surface, so it rolls back down', 'Pushing [something] with [something]', 'Opening [something]', 'Putting [something] on a surface', 'Taking [something] out of [something]', 'Spinning [something] that quickly stops spinning', 'Unfolding [something]', 'Moving [something] towards the camera', 'Putting [something] next to [something]', 'Scooping [something] up with [something]', 'Squeezing [something]', 'Failing to put [something] into [something] because [something] does not fit']

        # # ssv2_full
        # args.CLASS_NAME_TRAIN = ['Bending [something] until it breaks', 'Closing [something]', 'Covering [something] with [something]', 'Dropping [something] behind [something]', 'Dropping [something] in front of [something]', 'Dropping [something] into [something]', 'Folding [something]', 'Holding [something]', 'Holding [something] next to [something]', 'Letting [something] roll along a flat surface', 'Letting [something] roll down a slanted surface', 'Lifting a surface with [something] on it but not enough for it to slide down', 'Lifting [something] with [something] on it', 'Moving away from [something] with your camera', 'Moving [something] across a surface until it falls down', 'Moving [something] and [something] closer to each other', 'Moving [something] and [something] so they collide with each other', 'Moving [something] down', 'Moving [something] up', 'Plugging [something] into [something]', 'Poking a hole into [something soft]', "Poking [something] so lightly that it doesn't or almost doesn't move", 'Poking [something] so that it falls over', 'Pouring [something] into [something]', 'Pouring [something] into [something] until it overflows', 'Pouring [something] onto [something]', 'Pretending to be tearing [something that is not tearable]', 'Pretending to close [something] without actually closing it', 'Pretending to pick [something] up', 'Pretending to put [something] next to [something]', 'Pretending to spread air onto [something]', 'Pretending to take [something] out of [something]', 'Pulling [something] onto [something]', 'Pulling two ends of [something] so that it gets stretched', 'Pulling two ends of [something] so that it separates into two pieces', 'Pushing [something] from left to right', 'Pushing [something] off of [something]', 'Pushing [something] so that it falls off the table', 'Pushing [something] so that it slightly moves', 'Putting [number of] [something] onto [something]', 'Putting [something] and [something] on the table', "Putting [something] onto a slanted surface but it doesn't glide down", 'Putting [something] onto [something]', 'Putting [something similar to other things that are already on the table]', 'Showing a photo of [something] to the camera', 'Showing [something] behind [something]', '[Something] colliding with [something] and both are being deflected', 'Spilling [something] next to [something]', 'Spilling [something] onto [something]', 'Spinning [something] that quickly stops spinning', 'Spreading [something] onto [something]', 'Squeezing [something]', 'Stuffing [something] into [something]', 'Taking [something] from [somewhere]', 'Tearing [something] into two pieces', "Tilting [something] with [something] on it slightly so it doesn't fall down", 'Tilting [something] with [something] on it until it falls off', 'Tipping [something] with [something in it] over, so [something in it] falls out', 'Turning the camera downwards while filming [something]', 'Turning the camera left while filming [something]', 'Turning the camera upwards while filming [something]', 'Twisting (wringing) [something] wet until water comes out', 'Twisting [something]', 'Uncovering [something]']
        # args.CLASS_NAME_TEST = ['Approaching [something] with your camera', 'Digging [something] out of [something]', 'Dropping [something] next to [something]', 'Dropping [something] onto [something]', 'Failing to put [something] into [something] because [something] does not fit', 'Lifting up one end of [something] without letting it drop down', 'Picking [something] up', 'Poking a stack of [something] without the stack collapsing', 'Pouring [something] out of [something]', 'Pretending to open [something] without actually opening it', 'Pretending to put [something] behind [something]', 'Pretending to put [something] into [something]', 'Pretending to put [something] underneath [something]', 'Pretending to sprinkle air onto [something]', 'Pulling [something] from left to right', 'Pulling [something] out of [something]', 'Pushing [something] from right to left', 'Removing [something], revealing [something] behind', 'Showing [something] next to [something]', 'Showing that [something] is empty', 'Spilling [something] behind [something]', 'Taking [something] out of [something]', 'Throwing [something] in the air and letting it fall', 'Tipping [something] over']

        if args.scratch == "bc":
            args.scratch = "/mnt/storage/home2/tp8961/scratch"
        elif args.scratch == "bp":
            args.num_gpus = 4
            args.num_workers = 1
            args.scratch = "./"
        elif args.scratch == "new":
            args.scratch = "./imp_datasets/"

        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        if args.dataset == "ssv2":
            args.traintestlist = os.path.join(args.scratch, "splits/ssv2_CMN")
            # args.traintestlist = os.path.join(args.scratch, "splits/ssv2_OTAM")
            # args.path = "E:\Video_Recognition_DataSet\somethingsomethingv2_256x256q5_7l8.zip"
            args.path = "/media/data3/fsar/somethingsomethingv2_256x256q5_7l8.zip"
            # args.path = "/media/data3/fsar/ssv2_otam.zip"

        elif args.dataset == "kinetics":
            args.traintestlist = os.path.join(args.scratch, "splits/kinetics_CMN")
            args.path = "/media/data3/fsar/kinetics_CMN.zip"
        elif args.dataset == "ucf":
            args.traintestlist = os.path.join(args.scratch, "splits/ucf_ARN")
            # args.path = "E:\Video_Recognition_DataSet\somethingsomethingv2_256x256q5_7l8.zip"
            args.path = "/media/data3/fsar/ucf101_arn.zip"
        elif args.dataset == "hmdb":
            args.traintestlist = os.path.join(args.scratch, "splits/hmdb_ARN")
            # args.path = "E:\Video_Recognition_DataSet\HMDB51_ARN.zip"
            # args.path = os.path.join(args.scratch, "video_datasets/data/hmdb51_jpegs_256.zip")
            args.path = "/media/data1/ywc/fsar/hmdb51_arn.zip"
        elif args.dataset == "diving":
            args.traintestlist = os.path.join(args.scratch, "splits/diving_small")
            args.path = "/data0/fsar/Diving48_Small.zip"
            # args.traintestlist = os.path.join(args.scratch, "splits/diving_full")
            # args.path = "/data0/fsar/Diving48_Full.zip"

        with open("args.pkl", "wb") as f:
            pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            train_accuracies = []
            losses = []
            total_iterations = self.args.training_iterations

            iteration = self.start_iteration

            if self.args.test_model_only:
                print("Model being tested at path: " + self.args.test_model_path)
                self.load_checkpoint()
                accuracy_dict = self.test(session, 1)
                print(accuracy_dict)

            print('training....')
            for task_dict in self.video_loader:
                if iteration >= total_iterations:
                    break
                iteration += 1
                torch.set_grad_enabled(True)

                task_loss, task_accuracy = self.train_task(task_dict)
                train_accuracies.append(task_accuracy)
                losses.append(task_loss)

                # optimize
                if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):



                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.scheduler.step()
                if (iteration + 1) % self.args.print_freq == 0:
                    # print training stats
                    print_and_log(self.logfile, 'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                  .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                          torch.Tensor(train_accuracies).mean().item()))
                    train_logger.info(
                        "For Task: {0}, the training loss is {1} and Training Accuracy is {2}".format(iteration + 1,
                                                                                                      torch.Tensor(
                                                                                                          losses).mean().item(),
                                                                                                      torch.Tensor(
                                                                                                          train_accuracies).mean().item()))

                    avg_train_acc = torch.Tensor(train_accuracies).mean().item()
                    avg_train_loss = torch.Tensor(losses).mean().item()

                    train_accuracies = []
                    losses = []

                if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                    self.save_checkpoint(iteration + 1)

                if ((iteration + 1) in self.args.test_iters) and (iteration + 1) != total_iterations:
                    accuracy_dict = self.test(session, iteration + 1)
                    print(accuracy_dict)
                    self.test_accuracies.print(self.logfile, accuracy_dict)
                   

            # save the final model
            torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(
            task_dict)

        context_images = context_images.to(self.device).reshape((-1, self.args.seq_len) + context_images.size()[1:])
        context_labels = context_labels.to(self.device)
        target_images = target_images.to(self.device).reshape((-1, self.args.seq_len) + target_images.size()[1:])
        real_support, real_target, real_context = self.labels_batch_to_real(batch_class_list, target_labels, context_labels)


        model_dict = self.model(context_images, context_labels, target_images, real_context)


        target_logits = model_dict['logits'].to(self.device)

        target_labels = target_labels.to(self.device)

        task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch

        # Joint loss
        task_loss = task_loss 

        # Add the logits before computing the accuracy
        target_logits = target_logits

        task_accuracy = self.accuracy_fn(target_logits, target_labels)
        # text_task_accuracy = self.accuracy_fn(target_logits, target_labels)

        if math.isnan(task_loss):
            # logger.info(f"logits: {model_dict}")
            task_loss.backward(retain_graph=False)
            self.optimizer.zero_grad()
            return torch.tensor(0).type(task_loss.type()).to(self.device), torch.tensor(0).type(
                task_accuracy.type()).to(self.device)

        task_loss.backward(retain_graph=False)
        # task_accuracy = (task_accuracy + text_task_accuracy) / 2
        return task_loss, task_accuracy

    def test(self, session, num_episode):
        self.model.eval()
        with torch.no_grad():

            self.video_loader.dataset.train = False
            accuracy_dict = {}
            accuracies = []
            losses = []
            iteration = 0
            item = self.args.dataset
            for task_dict in self.video_loader:
                if iteration >= self.args.num_test_tasks:
                    break
                iteration += 1

                context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(
                    task_dict)
                context_images = context_images.to(self.device).reshape(
                    (-1, self.args.seq_len) + context_images.size()[1:])
                context_labels = context_labels.to(self.device)
                target_images = target_images.to(self.device).reshape(
                    (-1, self.args.seq_len) + target_images.size()[1:])
                
                real_support, real_target, real_context = self.labels_batch_to_real_test(batch_class_list, target_labels, context_labels)
                
                model_dict = self.model(context_images, context_labels, target_images, real_context)
                target_logits = model_dict['logits'].to(self.device)
                # text_target_logits = model_dict['text_logits'].to(self.device)


                target_labels = target_labels.to(self.device)

                # Add the logits before computing the accuracy
                target_logits = target_logits

                accuracy = self.accuracy_fn(target_logits, target_labels)


                loss = self.loss(target_logits, target_labels, self.device) / self.args.num_test_tasks
                # text_task_loss = self.loss(text_target_logits, target_labels, self.device) / self.args.tasks_per_batch


                # Joint loss
                loss = loss 

                eval_logger.info(
                    "For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(iteration + 1,
                                                                                                loss.item(),
                                                                                                accuracy.item()))
                losses.append(loss.item())
                accuracies.append(accuracy.item())
                del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            loss = np.array(losses).mean()
            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence, "loss": loss}
            eval_logger.info(
                "For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(num_episode, loss,
                                                                                            accuracy))

            self.video_loader.dataset.train = True
        self.model.train()

        return accuracy_dict

    def prepare_task(self, task_dict, images_to_device=True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
             'model_state_dict': self.model.state_dict(),
             'optimizer_state_dict': self.optimizer.state_dict(),
             'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        if self.args.test_model_only:
            checkpoint = torch.load(self.args.test_model_path)
        else:
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def labels_batch_to_real(self, batch_class_list, target_labels, context_labels):
            batch_class_list = batch_class_list.tolist()
            batch_class_list = list(map(int, batch_class_list))
            target_labels = target_labels.tolist()
            context_labels = context_labels.tolist()
            batch_real = {}
            real_support = []
            real_target = []
            real_context = []
            for i in range(len(batch_class_list)):
                batch_real[str(i)] = self.class_indict[str(batch_class_list[i])]
                real_support.append(self.class_indict[str(batch_class_list[i])])
            for i in range(len(target_labels)):
                real_target.append(batch_real[str(int(target_labels[i]))])
            for i in range(len(context_labels)):
                real_context.append(batch_real[str(int(context_labels[i]))])

            real_support = torch.FloatTensor(real_support).type(torch.LongTensor).to(self.device)
            real_target = torch.FloatTensor(real_target).type(torch.LongTensor).to(self.device)
            real_context = torch.FloatTensor(real_context).type(torch.LongTensor).to(self.device)
            return real_support, real_target, real_context

    def labels_batch_to_real_test(self, batch_class_list, target_labels, context_labels):
        batch_class_list = batch_class_list.tolist()
        batch_class_list = list(map(int, batch_class_list))
        target_labels = target_labels.tolist()
        context_labels = context_labels.tolist()
        batch_real = {}
        real_support = []
        real_target = []
        real_context = []
        for i in range(len(batch_class_list)):
            batch_real[str(i)] = self.class_indict_test[str(batch_class_list[i])]
            real_support.append(self.class_indict_test[str(batch_class_list[i])])
        for i in range(len(target_labels)):
            real_target.append(batch_real[str(int(target_labels[i]))])
        for i in range(len(context_labels)):
            real_context.append(batch_real[str(int(context_labels[i]))])

        real_support = torch.FloatTensor(real_support).type(torch.LongTensor).to(self.device)
        real_target = torch.FloatTensor(real_target).type(torch.LongTensor).to(self.device)
        real_context = torch.FloatTensor(real_context).type(torch.LongTensor).to(self.device)
        return real_support, real_target, real_context

if __name__ == "__main__":
    main()