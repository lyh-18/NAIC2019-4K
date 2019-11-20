import os
import glob

train_path = '/mnt/197_user/yhliu/DATA/REDS/train_sharp_bicubic/X4'
val_path = '/mnt/197_user/yhliu/DATA/REDS/val_sharp_bicubic/X4'

# mv the val set
val_folders = glob.glob(os.path.join(val_path, '*'))
#print(val_folders)
for folder in val_folders:
    new_folder_idx = '{:03d}'.format(int(folder.split('/')[-1]) + 240)
    #print(folder)
    #print(os.path.join(train_path, new_folder_idx))
    os.system('cp -r {} {}'.format(folder, os.path.join(train_path, new_folder_idx)))
