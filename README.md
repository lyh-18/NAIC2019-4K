# NAIC2019-4K
Xpixel Group project with NAIC2019-4K


# Reproduce instructions
## 使用docker构建训练/推理容器
1、拉取项目
```
    git clone https://github.com/lyh-18/NAIC2019-4K
```
2、进入项目主目录
```
    cd NAIC2019-4K
```
3、构建训练容器
```
    docker build -t AI4K:train -f Dockerfile_train .
```
4、运行训练容器
```
    docker run AI4K:train
```

5、构建推理容器  
(1) 指定输入的测试集目录及输出的切帧保存目录
编辑 extract_test_frames_demo.sh 文件
```
#!/bin/bash

folder="/tmp/data/test"   # please specify input .mp4 data folder
files=$(ls $folder)
save_folder="/tmp/data/testA_LR_png"  # please specify output .png data folder

for filename in $files
do 
 mkdir -p $save_folder/${filename%.*}
 ffmpeg -i $folder/$filename -r 30 $save_folder/${filename%.*}/%3d.png -y 
done
```
更改 folder 为输入的测试集mp4视频目录，save_folder 为输出的存放切帧后的png图片的目录。  
  
(2) 指定输入的测试集切帧图片目录及输出的合成视频目录
编辑 run_test.sh 文件  
```
#!/bin/bash

INPUT_DIR="/tmp/data/testA_LR_png/"
OUTPUT_FILE="/tmp/data/answer"

python test_video_no_GT_color_demo.py \
  --input_folder="${INPUT_DIR}" \
  --save_folder="${OUTPUT_FILE}" \
```
更改 INPUT_DIR 为输入的测试集切帧图片png目录，OUTPUT_FILE 为输出的存放mp4的目录。

(3) 构建推理容器
```
    docker build -t AI4K:test -f Dockerfile_test .
```
(4) 运行推理容器
```
    docker run AI4K:test
```

## 详细解读
## 1. 制作数据集
```
    sh prepare_data.sh
```
这一步请确保已经安装以下库：  
ffmpeg 4.2  
opencv-python  
matplotlib  
lmdb  
tqdm  
否则会报错。如果报错，可以定位到相应位置查看原因。具体每一个脚本可以 cat prepare_data.sh 查看后单独执行。

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
  制作lmdb (这一步会花费一些时间，执行该命令的同时，可以进行后面的训练超分网络步骤)
```
    python create_lmdb_train2_LR_corrected_demo.py
```

## 3. 训练超分网络
(0) 编译dcn模块  
进入 NAIC2019-4K/codes/models/archs/dcn/ 文件夹
```
    cd $HOME/NAIC2019-4K/codes/models/archs/dcn/
```
编译
```
    python setup.py develop
```

(1) 进入主目录的 codes 文件夹
```
    cd $HOME/NAIC2019-4K/codes
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
