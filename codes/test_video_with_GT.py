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
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    test_set = 'AI4K_val'    # Vid4 | YouKu10 | REDS4 | AI4K_val | zhibo | AI4K_val_bic
    test_name = 'PCD_Vis_Test_35_ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW_A01xxx_900000_AI4K_5000'                 #     'AI4K_val_Denoise_A02_420000'
    data_mode = 'sharp_bicubic'    # sharp_bicubic | blur_bicubic
    N_in = 5
    
    # load test set
    if test_set == 'Vid4':
        test_dataset_folder = '../datasets/Vid4/BIx4'
        GT_dataset_folder = '../datasets/Vid4/GT'
    elif test_set == 'YouKu10':
        test_dataset_folder = '../datasets/YouKu10/LR'
        GT_dataset_folder = '../datasets/YouKu10/HR'
    elif test_set == 'YouKu_val':
        test_dataset_folder = '/data0/yhliu/DATA/YouKuVid/valid/valid_lr_bmp'
        GT_dataset_folder = '/data0/yhliu/DATA/YouKuVid/valid/valid_hr_bmp'
    elif test_set == 'REDS4':
        test_dataset_folder = '../datasets/REDS4/{}'.format(data_mode)
        GT_dataset_folder = '../datasets/REDS4/GT'
    elif test_set == 'AI4K_val':
        test_dataset_folder = '/home/yhliu/AI4K/contest2/val2_LR_png/'
        GT_dataset_folder = '/home/yhliu/AI4K/contest1/val1_HR_png/'
    elif test_set == 'AI4K_val_bic':
        test_dataset_folder = '/home/yhliu/AI4K/contest1/val1_LR_png_bic/'
        GT_dataset_folder = '/home/yhliu/AI4K/contest1/val1_HR_png_bic/'
    elif test_set == 'zhibo':
        test_dataset_folder = '/data1/yhliu/SR_ZHIBO_VIDEO/Test_video_LR/'
        GT_dataset_folder = '/data1/yhliu/SR_ZHIBO_VIDEO/Test_video_HR/'
    
    flip_test = False
    
    #model_path = '../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth'
    #model_path = '../experiments/A01b/models/250000_G.pth'
    #model_path = '../experiments/A02_predenoise/models/415000_G.pth'
    model_path = '../experiments/A37_color_EDVR_35_220000_A01_5in_64f_10b_128_pretrain_A01xxx_900000_fix_before_pcd/models/5000_G.pth'



    predeblur, HR_in = False, False
    back_RBs = 10
    if data_mode == 'blur_bicubic':
        predeblur = True
    if data_mode == 'blur' or data_mode == 'blur_comp':
        predeblur, HR_in = True, True

    model = EDVR_arch.EDVR(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
    #model = my_EDVR_arch.MYEDVR(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
    #model = my_EDVR_arch.MYEDVR_RES(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)
    



    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = True #True | False

    save_folder = '../results/{}'.format(test_name)
    if test_set == 'zhibo':
        save_folder = '/data1/yhliu/SR_ZHIBO_VIDEO/SR_png_sample_150'
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
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    print(subfolder_l)
    print(subfolder_GT_l)
    #exit()
    
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        print(img_path_l)
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            #print(img_GT_path)
            img_GT_l.append(data_util.read_img(None, img_GT_path))
        #print(img_GT_l[0].shape)
        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).cpu()   #to(device)
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

            # calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])
            # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
            '''
            if data_mode == 'Vid4':  # bgr2y, [0, 1]
                GT = data_util.bgr2ycbcr(GT, only_y=True)
                output = data_util.bgr2ycbcr(output, only_y=True)
            '''

            output, GT = util.crop_border([output, GT], crop_border)
            crt_psnr = util.calculate_psnr(output * 255, GT * 255)
            logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                N_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_psnr_l,
                                                              avg_psnr_center_l, avg_psnr_border_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center,
                                                     psnr_border))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    main()
