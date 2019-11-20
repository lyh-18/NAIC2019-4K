#!/bin/bash

folder="SDR_540p_LR"
files=$(ls $folder)
save_folder="train1_LR_png"

for filename in $files
do
 #echo $folder/$filename
 #echo $save_folder/${filename%.*}
 
 mkdir -p $save_folder/${filename%.*}
 ffmpeg -i $folder/$filename -r 30 $save_folder/${filename%.*}/%3d.png -y
 
 
done
