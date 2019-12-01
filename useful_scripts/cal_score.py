
#import matplotlib.pyplot as plt
#import skimage.io as io
import cv2
from PIL import Image
import numpy as np
#import torch
import math
import time




def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim_my(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_my(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_my(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_my(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')





#filelist=[0.0,0.1,0.3,0.5,0.7]
#filelist=[1.0,1.3,1.6,2.0,2.5]
#filelist=[10.0]
ksizes=[25]
filelist=[5]

for ksize in ksizes:
    for name in filelist:
        list = [11553969, 17422120, 19524575, 21043777, 30063783, 30672625, 40267971, 47682563, 56318324, 70571139]
        # vmaf_EDVR_150000 = [65.27, 54.52, 75.45, 55.66, 51.72, 63.17, 56.67, 43.57, 35.23, 76.63]
        batch = "HF_{0}_{1}_AI4K_val_Denoise_A02_415000".format(str(ksize), str(name))
        HR_path = "/data1/public/DATA/add_ga/{0}/".format(batch)
        GT_path = "/data1/public/DATA/GT/"
        log_path = "/data1/public/log/add_ga_25_5_41500.txt"
        count = 0
        with open(log_path, "a+") as f:
            # f.write(str(i)+" psnr: "+str(result_psnr)+" SSIM: "+str(result_ssim)+" score: "+str(score)+"\n")
            f.write(batch + "\n")
            print(batch)
            starttime = time.time()
        for i in list:
            dir_HR = HR_path + str(i) + "/"
            dir_GT = GT_path + str(i) + "/"
            sum = 0
            sumss = 0
            for j in range(1, 101, 9):
                id = '%03d.png' % j
                id_G = '%03d.png' % j
                img_HR = Image.open(dir_HR + id)
                img_HR = np.array(img_HR.convert("RGB"))
                img_GT = Image.open(dir_GT + id_G)
                img_GT = np.array(img_GT.convert("RGB"))
                psrn = calculate_psnr(img_HR, img_GT)
                ssim = ssim_my(img_HR, img_GT)
                print(id, psrn)
                sum += psrn
                sumss += ssim
            result_psnr = sum / 12
            result_ssim = sumss / 12

            # score=0.25*(result_psnr/50)+0.25*((result_ssim-0.4)/0.6)+0.5*(vmaf_EDVR_150000[count]/80)
            count += 1

            # print(i,result_psnr,result_ssim,score)
            print(batch + str(i), result_psnr, result_ssim)
            with open(log_path, "a+") as f:
                # f.write(str(i)+" psnr: "+str(result_psnr)+" SSIM: "+str(result_ssim)+" score: "+str(score)+"\n")
                f.write(str(i) + " psnr: " + str(result_psnr) + " ssim:" + str(result_ssim) + "\n")

        endtime = time.time()
        dtime = endtime - starttime
        print(batch, dtime)





