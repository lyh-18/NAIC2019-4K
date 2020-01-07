import time
import math
import numpy as np
import torch
import torchvision.utils
from data import create_dataloader, create_dataset
from utils import util
import sys

import torch.multiprocessing as mp
# mp.set_start_method('spawn')

opt = {}

opt['name'] = 'AI4K_Recurr'
opt['dataroot_GT'] = '/mnt/lustre/share/kckchan/Datasets/AI4K/train1_HR_png'
opt['dataroot_LQ'] = '/mnt/lustre/share/kckchan/Datasets/AI4K/train1_LR_png'

opt['mode'] = 'AI4K_Recurr'
opt['N_frames'] = 10
opt['phase'] = 'train'  # 'train' | 'val'
opt['video_class'] = 'all'
opt['use_shuffle'] = True
opt['n_workers'] = 0
opt['batch_size'] = 1
opt['LQ_size'] = 64
opt['GT_size'] = 256

opt['scale'] = 4
opt['use_flip'] = True
opt['use_rot'] = True
opt['interval_list'] = [3]
opt['random_reverse'] = False
opt['cache_keys'] = 'REDS_trainval_keys.pkl'
opt['data_type'] = 'img'  # img | lmdb | mc | ceph
util.set_random_seed(0)
util.mkdir('tmp')
train_set = create_dataset(opt)
opt['dist'] = False
opt['gpu_ids'] = [0]
train_loader = create_dataloader(train_set, opt, opt, None)
nrow = int(math.sqrt(opt['batch_size']))
# if opt['phase'] == 'train':
#     padding = 2
# else:
#     padding = 0
start_time = time.time()
print('start...')
for i, data in enumerate(train_loader):
    # test dataloader time
    # if i == 1:
    #     start_time = time.time()
    # if i == 500:
    #     print(time.time() - start_time)
    #     break
    if i > 5:
        break
    LRs = data['LQs']
    GTs = data['GT']
    GTs_bic4x = data['img_GT_bic4x']
    key = data['key']

    print(LRs.size())
    print(GTs.size())
    print(GTs_bic4x.size())
    raise NotImplementedError
    # print(i)

    # save LR images
    for j in range(LRs.size(1)):
        torchvision.utils.save_image(LRs[:, j, :, :, :], 'tmp/LR_{:03d}_{}.png'.format(i, j),
                                     nrow=nrow, padding=2, normalize=False)
        torchvision.utils.save_image(GTs[:, j, :, :, :], 'tmp/GT_{:03d}_{}.png'.format(i, j),
                                     nrow=nrow, padding=2, normalize=False)
    # raise NotImplementedError
