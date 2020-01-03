# NAIC2019-4K
Xpixel Group project with NAIC2019-4K


# Reproduce instructions
## 1. 制作数据集
```
    sh prepare_data.sh
```

## 2. 训练调色网络
(1) 进入 ColorNet/codes 文件夹
```
    cd ColorNet/codes
```

(2) 执行训练命令 (训练过程大约30h)
```
    python train_color_demo.py -opt options/train/train_ColorNet_demo.yml
```

(3) 使用训练好的调色模型对复赛 LR 进行调色
```
    python test_color_vid_demo.py -opt options/test/test_ColorNet_Train2_demo.yml
```

(4) 制作新的复赛 LR 数据集  
  回到项目主目录
```
    cd ../..
```
  制作lmdb
```
    python create_lmdb_train2_LR_corrected_demo.py
```

## 3. 训练超分网络
(0) 编译dcn模块
进入 NAIC2019-4K/codes/models/archs/dcn/ 文件夹
```
    cd NAIC2019-4K/codes/models/archs/dcn/
```
编译
```
    python setup.py develop
```

(1) 进入主目录的 codes 文件夹
```
    cd NAIC2019-4K/codes
```

(2) 使用初赛数据集预训练超分网络 (训练过程大约30h)
```
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train_demo.py -opt options/train/train_EDVR_AI4K_demo.yml --launcher pytorch
```

(3) 使用颜色校准后的复赛数据集继续训练超分网络 (训练过程大约20h)
```
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train_demo.py -opt options/train/train_EDVR_AI4K_color_demo.yml --launcher pytorch
```

## 4. 预测及合成视频
(1) 使用训练好的模型对测试集进行预测，生成png图片 (50个视频大约需要3h)
```
    python test_video_no_GT_color_demo.py
```
(2) 合成视频 (仍在codes文件夹下)
```
    sh synthesize_mp4.sh
```
最终输出的视频保存在 /tmp/data/answer 文件夹中
