import os
import math
import argparse
import random
import logging
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')


args = parser.parse_args()
opt = option.parse(args.opt, is_train=True)


opt = option.dict_to_nonedict(opt)



model = create_model(opt)