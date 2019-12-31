#!/bin/bash

folder="/tmp/data/train_1st/4K"
files=$(ls $folder)
save_folder="/tmp/data/train1_HR_png"

for filename in $files
do
 #echo $folder/$filename
 #echo $save_folder/${filename%.*}
 
 mkdir -p $save_folder/${filename%.*}
 ffmpeg -i $folder/$filename -r 24000/1001 $save_folder/${filename%.*}/%3d.png -y
 
 
done
