import os
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

exp_name="AI4K_val_Denoise_A02_415000"
input_folder = '/data1/public/DATA/{0}'.format(exp_name)
alpha_list = [5]
k_sizes = [25]
correct_mean_var=0

input_video_name_list = os.listdir(input_folder)
input_video_name_list.sort()


def generate_add_HF(alpha, k_size, save_folder):
    for video_name in tqdm(input_video_name_list):
        if video_name<"30063783" :
            video_path = os.path.join(input_folder, video_name)
            img_name_list = os.listdir(video_path)
            img_name_list.sort()
            # print(img_name_list)
            num=0
            for img_name in img_name_list:
                num+=1
                if num<110:

                    img_path = os.path.join(video_path, img_name)
                    # print(img_path)
                    img = cv2.imread(img_path)

                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # --------------------------------------------------------
                    img_blur = cv2.GaussianBlur(img, (k_size, k_size), 0)

                    img_copy = img.copy()
                    img_copy = img_copy.astype(np.float)
                    img_blur_copy = img_blur.copy()
                    img_blur_copy = img_blur_copy.astype(np.float)

                    img_res = img_copy - img_blur_copy  # cv2.subtract(img, img_blur) #
                    img_add = img_copy + alpha * img_res
                    # ----------------------------------------------------------
                    #img_add=img+img*float(alpha)/1000
                    if correct_mean_var:
                        mean_l = []
                        std_l = []
                        for j in range(3):
                            mean_l.append(np.mean(img[:, :, j]))
                            std_l.append(np.std(img[:, :, j]))
                        for j in range(3):
                            # correct twice
                            mean = np.mean(img_add[:, :, j])
                            img_add[:, :, j] = img_add[:, :, j] - mean + mean_l[j]
                            std = np.std(img_add[:, :, j])
                            img_add[:, :, j] = img_add[:, :, j] / std * std_l[j]

                            mean = np.mean(img_add[:, :, j])
                            img_add[:, :, j] = img_add[:, :, j] - mean + mean_l[j]
                            std = np.std(img_add[:, :, j])
                            img_add[:, :, j] = img_add[:, :, j] / std * std_l[j]

                    img_add = np.clip(img_add, 0, 255).astype(np.uint8)
                    #img_res = np.clip(img_res*20, 0, 255).astype(np.uint8)

                    '''
                    plt.imshow(img)
                    plt.show()
                    #plt.imshow(img_res)
                    #plt.show()
                    plt.imshow(img_add)
                    plt.show()
                    break
                    '''

                    save_path = os.path.join(save_folder, video_name)

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, img_name), img_add)
                    #cv2.imwrite(os.path.join(save_path, "o"+img_name), img)


for k_size in k_sizes:
    for alpha in alpha_list:
        print('Processing: alpha={}, k_size={}'.format(alpha, k_size))
        save_folder = 'HF_' + str(k_size) + '_' + str(alpha) + '_' + exp_name
        #save_folder = "RES20"+'HF_' + str(k_size) + '_' + str(alpha) + '_' + exp_name
        generate_add_HF(alpha, k_size, save_folder)
