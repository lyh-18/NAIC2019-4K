import os
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm


input_folder = 'val1_LR_png'
save_folder = 'val1_LR_png_bic'



input_video_name_list = os.listdir(input_folder)
input_video_name_list.sort()

#print(input_video_name_list)

for video_name in tqdm(input_video_name_list):
    video_path = os.path.join(input_folder, video_name)
    img_name_list = os.listdir(video_path)
    img_name_list.sort()
    #print(img_name_list)
    for img_name in img_name_list:
        img_path = os.path.join(video_path, img_name)
        #print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), cv2.INTER_CUBIC)
        
        '''
        plt.imshow(img)
        plt.show()
        plt.imshow(img_add)
        plt.show()
        '''
        save_path = os.path.join(save_folder, video_name)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, img_name), img)
        
        
    
        