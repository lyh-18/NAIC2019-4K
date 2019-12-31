import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import math
import pdb

#################################################################################


LR_input_folder = '/tmp/data/val2_LR_png'
HR_input_folder = '/tmp/data/val1_HR_png'
new_LR_folder = '/tmp/data/new_val_v2/new_LR'
new_HR_folder = '/tmp/data/new_val_v2/new_HR'

# pdb.set_trace()

input_video_name_dict = {'11553969': ['004.png', '009.png', '011.png', '019.png', '021.png', '022.png', '025.png', '027.png', '029.png', 
'030.png', '047.png', '050.png', '062.png', '071.png', '084.png', '089.png', '093.png', '095.png', '097.png', '099.png'],

'17422120': ['001.png', '009.png', '010.png', '015.png', '020.png', '025.png', '030.png', '035.png', '040.png', '045.png', '050.png', 
'055.png', '060.png', '065.png', '070.png', '080.png', '085.png', '090.png', '095.png', '100.png'],

'19524575': ['001.png', '005.png', '009.png', '013.png', '017.png', '021.png', '025.png', '029.png', '033.png', '037.png', '041.png', 
'045.png', '049.png', '053.png', '057.png', '058.png', '059.png', '061.png', '063.png', '064.png', '067.png', '071.png', 
'075.png', '079.png', '083.png', '087.png', '088.png', '090.png', '095.png', '100.png'],

'21043777': ['001.png', '002.png', '006.png', '007.png', '011.png', '012.png', '016.png', '020.png', '024.png', '028.png', '032.png', 
'036.png', '040.png', '044.png', '048.png', '052.png', '056.png', '060.png', '061.png', '063.png', '067.png', '071.png', 
'075.png', '079.png', '083.png', '087.png', '091.png', '092.png', '096.png', '100.png'],

'30063783': ['001.png', '011.png', '016.png', '021.png', '026.png', '031.png', '036.png', '041.png', '047.png', '052.png', '057.png', 
'058.png', '063.png', '068.png', '073.png', '078.png', '086.png', '087.png', '092.png', '100.png'],

'30672625': ['001.png', '011.png', '012.png', '014.png', '017.png', '025.png', '026.png', '034.png', '050.png', '056.png', '060.png', 
'063.png', '066.png', '071.png', '072.png', '077.png', '082.png', '087.png', '094.png', '100.png'],

'40267971': ['001.png', '002.png', '004.png', '006.png', '007.png', '009.png', '011.png', '012.png', '013.png', '014.png', '016.png', 
'017.png', '018.png', '019.png', '020.png', '021.png', '023.png', '024.png', '026.png', '028.png', '029.png', '030.png', 
'031.png', '032.png', '034.png', '035.png', '036.png', '038.png', '039.png', '040.png', '042.png', '043.png', '044.png', 
'046.png', '047.png', '048.png', '050.png', '051.png', '052.png', '054.png', '055.png', '056.png', '058.png', '059.png', 
'060.png', '063.png', '066.png', '067.png', '068.png', '070.png', '071.png', '072.png', '074.png', '075.png', '076.png', 
'078.png', '079.png', '080.png', '082.png', '083.png', '084.png', '086.png', '087.png', '088.png', '090.png', '092.png', 
'094.png', '096.png', '098.png', '100.png'],

'47682563': ['001.png', '006.png', '011.png', '013.png', '014.png', '019.png', '024.png', '031.png', '032.png', '037.png', '042.png', 
'047.png', '052.png', '057.png', '062.png', '072.png', '086.png', '087.png', '095.png', '100.png'],

'56318324': ['001.png', '006.png', '011.png', '021.png', '026.png', '036.png', '041.png', '046.png', '051.png', '056.png', '061.png', 
'066.png', '071.png', '076.png', '081.png', '086.png', '091.png', '094.png', '095.png', '100.png'],

'70571139': ['001.png', '006.png', '011.png', '016.png', '020.png', '025.png', '030.png', '035.png', '040.png', '045.png', '050.png', 
'055.png', '060.png', '065.png', '070.png', '075.png', '085.png', '090.png', '095.png', '100.png']}

if not os.path.exists(new_LR_folder):
    os.makedirs(new_LR_folder)

if not os.path.exists(new_HR_folder):
    os.makedirs(new_HR_folder)


for video_name in tqdm(input_video_name_dict.keys()):

    HR_video_path = os.path.join(HR_input_folder, video_name)
    LR_video_path = os.path.join(LR_input_folder, video_name)

    # for img_name in tqdm(img_name_list):
    for img_name in input_video_name_dict[video_name]:

        HR_img_path = os.path.join(HR_video_path, img_name)
        LR_img_path = os.path.join(LR_video_path, img_name)
        
        HR_img = cv2.imread(HR_img_path)
        LR_img = cv2.imread(LR_img_path)

        save_name = '{}_{}'.format(video_name, img_name)

        cv2.imwrite(os.path.join(new_HR_folder, save_name), HR_img)
        cv2.imwrite(os.path.join(new_LR_folder, save_name), LR_img)

