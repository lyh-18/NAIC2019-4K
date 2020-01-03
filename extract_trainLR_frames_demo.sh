#!/bin/bash

folder="/tmp/data/train_1st/540p"
files=$(ls $folder)
save_folder="/tmp/data/train1_LR_png"

for filename in $files
do
 #echo $folder/$filename
 #echo $save_folder/${filename%.*}
 
 mkdir -p $save_folder/${filename%.*}
 ffmpeg -i $folder/$filename -r 30 $save_folder/${filename%.*}/%3d.png -y
 
 
done



folder="/tmp/data/train_2nd/540p"
files=$(ls $folder)
save_folder="/tmp/data/train2_LR_png"

for filename in $files
do
 #echo $folder/$filename
 #echo $save_folder/${filename%.*}
 
 mkdir -p $save_folder/${filename%.*}
 ffmpeg -i $folder/$filename -r 24000/1001 $save_folder/${filename%.*}/%3d.png -y
 
 
done