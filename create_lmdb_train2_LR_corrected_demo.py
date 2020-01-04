"""Create lmdb files for [General images (291 images/DIV2K) | Vimeo90K | REDS] training datasets"""

import sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import codes.data.util as data_util  # noqa: E402
import codes.utils.util as util  # noqa: E402


def main():
    dataset = 'AI4K' # AI4K
    mode = 'LR'  # GT | LR

    if dataset == 'AI4K':
        VideoSR(mode)
    elif dataset == 'test':
        test_lmdb('/data0/yhliu/AI4K/contest1/train1_LR.lmdb', 'ai4k')


def read_image_worker(path, key):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)


def VideoSR(mode):
    """Create lmdb for the Video dataset, each image with a fixed size
    LR: [3, 540, 960], key: 000_00000000
    GT: [3, 2160, 3840], key: 000_00000000
    key: 000_00000000

    """
    #### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 2000 #5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    if mode == 'GT':
        img_folder = '/tmp/data/train1_HR_png'
        lmdb_save_path = '/tmp/data/train1_HR.lmdb'
        H_dst, W_dst = 2160, 3840
    elif mode == 'LR':
        img_folder = './ColorNet/results/trainLR_Reproduce_ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW_re_100k/correted_train2_LR_png'
        lmdb_save_path = '/tmp/data/train2_LR_corrected.lmdb'
        #img_folder = '/home/yhliu/BasicSR/results/trainLR_35_ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW_re_100k_220000/trainLR_35_ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW_re_100k_220000/'
        #lmdb_save_path = '/home/yhliu/AI4K/contest2/train2_LR_35_220000.lmdb'
        
        H_dst, W_dst = 540, 960
    
    n_thread = 40 #40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = data_util._get_paths_from_images(img_folder)
    keys = []
    for img_path in all_img_list:
        split_rlt = img_path.split('/')
        folder = split_rlt[-2]
        img_name = split_rlt[-1].split('.png')[0]
        keys.append(folder + '_' + img_name)

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if 'flow' in mode:
            H, W = data.shape
            assert H == H_dst and W == W_dst, 'different shape.'
        else:
            H, W, C = data.shape
            assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    meta_info['name'] = 'AI4K_{}_train1'.format(mode)
    channel = 1 if 'flow' in mode else 3
    meta_info['resolution'] = '{}_{}_{}'.format(channel, H_dst, W_dst)
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def test_lmdb(dataroot, dataset='REDS'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(osp.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))
    print('# keys: ', meta_info['keys'][10])
    # read one image
    
    if dataset == 'vimeo90k':
        key = '00001_0001_4'
    elif dataset == 'ai4k':
        key = '80293858_004'
    else:
        key = '000_00000000'
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = [int(s) for s in meta_info['resolution'].split('_')]
    img = img_flat.reshape(H, W, C)
    cv2.imwrite('test.png', img)
    


if __name__ == "__main__":
    main()
