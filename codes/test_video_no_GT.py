'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch
import models.archs.my_EDVR_arch as my_EDVR_arch

import time

def main():
    #################
    # configurations
    #################
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True
    
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    test_set = 'AI4K_test'    # Vid4 | YouKu10 | REDS4 | AI4K_test
    data_mode = 'sharp_bicubic'    # sharp_bicubic | blur_bicubic
    test_name = 'AI4K_TEST_Denoise_A02_415000_flip'             #'AI4K_TEST_Denoise_A02_265000'    |  AI4K_test_A01b_145000
    N_in = 5
    
    # load test set
    if test_set == 'AI4K_test':
        test_dataset_folder = '/home/yhliu/AI4K/test1_LR_png/'
    
    flip_test = True  #False
    
    #model_path = '../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth'
    #model_path = '../experiments/002_EDVR_EDVRwoTSAIni_lr4e-4_600k_REDS_LrCAR4S_fixTSA50k_new/models/latest_G.pth'
    #model_path = '../experiments/A01b/models/145000_G.pth'
    model_path = '../experiments/A02_predenoise/models/415000_G.pth'
    



    predeblur, HR_in = False, False
    back_RBs = 10
    if data_mode == 'blur_bicubic':
        predeblur = True
    if data_mode == 'blur' or data_mode == 'blur_comp':
        predeblur, HR_in = True, True

    #model = EDVR_arch.EDVR(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
    model = my_EDVR_arch.MYEDVR(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)



    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = True

    save_folder = '../results/{}'.format(test_name)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    model=nn.DataParallel(model)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    #print(subfolder_l)
    #print(subfolder_GT_l)
    #exit()
    
    # for each subfolder
    for subfolder in subfolder_l:
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        #print(img_path_l)
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).cpu()
            print(imgs_in.size())

            if flip_test:
                output = util.flipx4_forward(model, imgs_in)
            else:
                start_time = time.time()
                output = util.single_forward(model, imgs_in)
                end_time = time.time()
                print('Forward One image:', end_time-start_time)
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)
            
            logger.info('{:3d} - {:25}'.format(img_idx + 1, img_name))



    logger.info('################ Tidy Outputs ################')
    
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Total execution time: ',end-start)
