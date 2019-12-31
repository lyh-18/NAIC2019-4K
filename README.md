# NAIC2019-4K
Xpixel Group project with NAIC2019-4K

文本
------
### 普通文本
这是一段普通的文本
### 单行文本
    Hello,大家好，我是果冻虾仁。
在一行开头加入1个Tab或者4个空格。
### 文本块

# Reproduce instructions
## 1 制作数据集
(1) 抽取 HR 帧  
    sh extract_trainHR_frames_demo.sh  
(2) 抽取 LR 帧  
    sh extract_trainLR_frames_demo.sh  
(3) 抽取测试集 LR 帧  
    sh extract_test_frames_demo.sh  
(4) 抽取训练过程验证集  
    sh get_val10_demo.sh  
