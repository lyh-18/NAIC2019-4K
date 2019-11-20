"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.util import ProgressBar  # noqa: E402
import data.util as data_util  # noqa: E402



global_idx = 0
abandon_list = []
LR = False

def main():
    mode = 'pair'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
    opt = {}
    opt['n_thread'] = 10
    opt['compression_level'] = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.
    if mode == 'single':
        opt['input_folder'] = '../../datasets/DIV2K/DIV2K_train_HR'
        opt['save_folder'] = '../../datasets/DIV2K/DIV2K800_sub'
        opt['crop_sz'] = 480  # the size of each sub-image
        opt['step'] = 240  # step of the sliding crop window
        opt['thres_sz'] = 48  # size threshold
        extract_signle(opt)
    elif mode == 'pair':
        GT_folder = '/home/yhliu/AI4K/contest1/train1_HR_png/'
        LR_folder = '/home/yhliu/AI4K/contest1/train1_LR_png/'
        save_GT_folder = '/data0/yhliu/AI4K/contest1/train1_HR_png_sub/'
        save_LR_folder = '/data0/yhliu/AI4K/contest1/train1_LR_png_sub/'
        scale_ratio = 4
        crop_sz = 480  # the size of each sub-image (GT)
        step = 240  # step of the sliding crop window (GT)
        thres_sz = 48  # size threshold
        ########################################################################
        # check that all the GT and LR images have correct scale ratio
        img_GT_list = data_util._get_paths_from_images(GT_folder)        
        img_LR_list = data_util._get_paths_from_images(LR_folder)
        print(len(img_GT_list))
        print(len(img_GT_list))
        assert len(img_GT_list) == len(img_LR_list), 'different length of GT_folder and LR_folder.'
        '''
        for path_GT, path_LR in zip(img_GT_list, img_LR_list):
            img_GT = Image.open(path_GT)
            img_LR = Image.open(path_LR)
            w_GT, h_GT = img_GT.size
            w_LR, h_LR = img_LR.size
            assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
                w_GT, scale_ratio, w_LR, path_GT)
            assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
                w_GT, scale_ratio, w_LR, path_GT)
        '''
        # check crop size, step and threshold size
        assert crop_sz % scale_ratio == 0, 'crop size is not {:d}X multiplication.'.format(
            scale_ratio)
        assert step % scale_ratio == 0, 'step is not {:d}X multiplication.'.format(scale_ratio)
        assert thres_sz % scale_ratio == 0, 'thres_sz is not {:d}X multiplication.'.format(
            scale_ratio)
        print('process GT...')
        
        opt['GT_input_folder'] = GT_folder
        opt['GT_save_folder'] = save_GT_folder
        opt['GT_crop_sz'] = crop_sz
        opt['GT_step'] = step
        opt['GT_thres_sz'] = thres_sz
        
        opt['LR_input_folder'] = LR_folder
        opt['LR_save_folder'] = save_LR_folder
        opt['LR_crop_sz'] = crop_sz // scale_ratio
        opt['LR_step'] = step // scale_ratio
        opt['LR_thres_sz'] = thres_sz // scale_ratio
        
        clip_folders = os.listdir(GT_folder)
        clip_folders.sort()
        clip_folders = clip_folders[400:600]
        
        print(len(clip_folders))
        for one_clip in clip_folders:
            opt['clip_folder'] = one_clip           
            GT_img_list = data_util._get_paths_from_images(os.path.join(GT_folder, one_clip))
            LR_img_list = data_util._get_paths_from_images(os.path.join(LR_folder, one_clip))
            for GT_path, LR_path in zip(GT_img_list, LR_img_list):
                opt['subclip_folder'] = GT_path.split('/')[-1].split('.')[0]
                print(GT_path,LR_path)
                extract_signle(opt, GT_path, LR_path)
        
                
            
        
            
        
        
        
        assert len(data_util._get_paths_from_images(save_GT_folder)) == len(
            data_util._get_paths_from_images(
                save_LR_folder)), 'different length of save_GT_folder and save_LR_folder.'
    else:
        raise ValueError('Wrong mode.')


def extract_signle(opt, GT_path, LR_path):
    #GT_input_folder = os.path.join(opt['GT_input_folder'],opt['clip_folder'])
    #print(input_folder)
    
    
    
        
    
    worker(GT_path, LR_path, opt)
    #def update(arg):
    #    pbar.update(arg)

    #pbar = ProgressBar(690)

    #pool = Pool(opt['n_thread'])
    
    #pool.apply_async(worker, args=(GT_path, LR_path, opt), callback=update)
    #pool.close()
    #pool.join()
    print('All subprocesses done.')


def worker(GT_path, LR_path, opt):
    
    global_idx = 0
    abandon_list = []
    # HR
    crop_sz = opt['GT_crop_sz']
    step = opt['GT_step']
    thres_sz = opt['GT_thres_sz']
    img_name = osp.basename(GT_path)
    img = cv2.imread(GT_path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)
    
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
                        
            GT_save_folder = os.path.join(opt['GT_save_folder'], opt['clip_folder'], 's{:03d}'.format(index))
            if not osp.exists(GT_save_folder):
                os.makedirs(GT_save_folder)
                #print('mkdir [{:s}] ...'.format(GT_save_folder))
            '''
            else:
                print('Folder [{:s}] already exists. Exit...'.format(GT_save_folder))
                sys.exit(1)
            '''
    
            
            
            
            
            
            cv2.imwrite(
                osp.join(GT_save_folder,
                         img_name.replace('.png', '_s{:03d}.png'.format(index))), crop_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
            
            
            
    
    global_idx = 0
    # LR
    crop_sz = opt['LR_crop_sz']
    step = opt['LR_step']
    thres_sz = opt['LR_thres_sz']
    img_name = osp.basename(LR_path)
    img = cv2.imread(LR_path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            
            LR_save_folder = os.path.join(opt['LR_save_folder'], opt['clip_folder'], 's{:03d}'.format(index))
            if not osp.exists(LR_save_folder):
                os.makedirs(LR_save_folder)
                #print('mkdir [{:s}] ...'.format(LR_save_folder))
            '''
            else:
                print('Folder [{:s}] already exists. Exit...'.format(LR_save_folder))
                sys.exit(1)
            '''
            
            cv2.imwrite(
                osp.join(LR_save_folder,
                         img_name.replace('.png', '_s{:03d}.png'.format(index))), crop_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
            
    
    
    
    
    return 'Processing {:s} ...'.format(img_name)




    


if __name__ == '__main__':
    main()
