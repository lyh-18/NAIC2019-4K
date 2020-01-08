import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger('base')


class AI4KDataset(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(AI4KDataset, self).__init__()
        self.opt = opt
        if opt['video_class']:
            self.video_class = opt['video_class']  # all | movie | cartoon | lego
        else:
            self.video_class = 'all'

        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        # self.half_N_frames = opt['N_frames'] // 2
        self.N_frames = opt['N_frames']
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            train_list = []
            if self.video_class == 'all':
                pass
            elif self.video_class == 'movie':
                with open('data/movie_list.txt', 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        train_list.append(line)
                    #print((train_list))
                for item in self.paths_GT.copy():
                    if item.split('_')[0] not in train_list:
                        self.paths_GT.remove(item)
            elif self.video_class == 'cartoon':
                with open('data/cartoon_list.txt', 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        train_list.append(line)
                for item in self.paths_GT.copy():
                    if item.split('_')[0] not in train_list:
                        self.paths_GT.remove(item)
            elif self.video_class == 'lego':
                with open('data/lego_list.txt', 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        train_list.append(line)
                for item in self.paths_GT.copy():
                    if item.split('_')[0] not in train_list:
                        self.paths_GT.remove(item)

            # clear bad data
            for item in self.paths_GT.copy():
                if item.split('_')[0] == '15922480':
                    self.paths_GT.remove(item)

            logger.info('Using lmdb meta info for cache keys.')
        elif self.data_type == 'img':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])

        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        #else:
        #    raise ValueError(
        #        'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        assert self.paths_GT, 'Error: GT path is empty.'
        #rint((self.paths_GT))
        # print(len(self.paths_GT))

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        if self.data_type == 'lmdb':
            key = self.paths_GT[index]
            name_a, name_b = key.split('_')

        elif self.data_type == 'img':
            key = self.paths_GT[index]
            name_a = key.split('/')[-2]
            name_b = key.split('/')[-1].split('.')[0]

        first_frame_idx = int(name_b)

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        while first_frame_idx + (self.N_frames - 1) * interval > 100:
            first_frame_idx = random.randint(1, 100)
        neighbor_list = list(
            range(first_frame_idx, first_frame_idx + self.N_frames * interval, interval))
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.N_frames, 'Wrong length of neighbor list: {}'.format(
            len(neighbor_list))

        #### get LQ images
        LQ_size_tuple = (3, 540, 960) if self.LR_input else (3, 2160, 3840)
        GT_size_tuple = (3, 2160, 3840)

        img_LQ_l = []
        img_GT_l = []
        for v in neighbor_list:
            img_LQ_path = osp.join(self.LQ_root, name_a, '{:03d}.png'.format(v))
            img_GT_path = osp.join(self.GT_root, name_a, '{:03d}.png'.format(v))

            if self.data_type == 'mc':
                # LQ
                if self.LR_input:
                    img_LQ = self._read_img_mc(img_LQ_path)
                else:
                    img_LQ = self._read_img_mc_BGR(self.LQ_root, name_a, '{:03d}'.format(v))
                img_LQ = img_LQ.astype(np.float32) / 255.
                # GT
                img_GT = self._read_img_mc_BGR(self.GT_root, name_a, '{:03d}'.format(v))
            elif self.data_type == 'lmdb':
                # LQ
                img_LQ = util.read_img(self.LQ_env, '{}_{:03d}'.format(name_a, v), LQ_size_tuple)
                # GT
                img_GT = util.read_img(self.GT_env, '{}_{:03d}'.format(name_a, v), GT_size_tuple)
            else:
                # LQ
                img_LQ = util.read_img(None, img_LQ_path)
                # GT
                img_GT = util.read_img(None, img_GT_path)
            img_LQ_l.append(img_LQ)
            img_GT_l.append(img_GT)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale

                # choose patch whose variance is larger than threshod
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_GT_l = [
                    v[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in img_GT_l
                ]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_GT_l]
            # augmentation - flip, rotate

            img_LQ_l += img_GT_l
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:self.N_frames]
            img_GT_l = rlt[self.N_frames:]

        img_GT_bic4x_l = [util.imresize_np(v.copy(), 1 / 4, True) for v in img_GT_l]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        img_GTs = np.stack(img_GT_l, axis=0)
        img_GTs_bic4x = np.stack(img_GT_bic4x_l, axis=0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GTs = img_GTs[:, :, :, [2, 1, 0]]
        img_GTs_bic4x = img_GTs_bic4x[:, :, :, [2, 1, 0]]
        '''
        import matplotlib.pyplot as plt
        plt.subplot(2,2,1)
        plt.imshow(img_GT)
        plt.subplot(2,2,2)
        plt.imshow(img_LQs[2,:,...])
        plt.subplot(2,2,3)
        plt.imshow(img_GT_bic4x)
        plt.subplot(2,2,4)
        plt.imshow(img_GT_bic4x-img_LQs[2,:,...])
        plt.show()
        exit()
        '''

        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs,
                                                                     (0, 3, 1, 2)))).float()
        img_GTs_bic4x = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GTs_bic4x, (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GTs, 'img_GT_bic4x': img_GTs_bic4x, 'key': key}

    def __len__(self):
        return len(self.paths_GT)
