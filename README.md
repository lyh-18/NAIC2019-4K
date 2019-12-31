# NAIC2019-4K
Xpixel Group project with NAIC2019-4K


# Reproduce instructions
## 1 制作数据集
```
    sh prepare_data.sh
```

## 2 训练调色网络
(1) 进入 ColorNet/codes 文件夹
```
    cd ColorNet/codes
```

(2) 执行训练命令 (训练过程大约10h)
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
