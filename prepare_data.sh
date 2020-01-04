#!/bin/bash

sh extract_trainHR_frames_demo.sh
sh extract_trainLR_frames_demo.sh

sh get_val10_demo.sh
python3 get_video_bic_demo.py
python3 create_color_val_demo.py

python3 create_lmdb_train1_LR_demo.py
python3 create_lmdb_train2_LR_demo.py
python3 create_lmdb_train1_HR_demo.py
