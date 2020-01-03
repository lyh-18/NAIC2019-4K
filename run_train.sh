#!/bin/bash

sh prepare_data.sh

cd ColorNet/codes 
python train_color_demo.py -opt options/train/train_ColorNet_demo.yml 
python test_color_vid_demo.py -opt options/test/test_ColorNet_Train2_demo.yml 
    
cd ../.. 
python create_lmdb_train2_LR_corrected_demo.py
    
cd codes/models/archs/dcn/ 
python setup.py develop 
cd ../../..
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train_demo.py -opt options/train/train_EDVR_AI4K_demo.yml --launcher pytorch 
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train_demo.py -opt options/train/train_EDVR_AI4K_color_demo.yml --launcher pytorch 