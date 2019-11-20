#!/bin/bash

folder="SDR_4K_HR"
files=$(ls $folder)
save_folder="train1_HR_png"

for filename in $files
do
 #echo $folder/$filename
 #echo $save_folder/${filename%.*}
 
 mkdir -p $save_folder/${filename%.*}
 ffmpeg -i $folder/$filename -r 23.98 $save_folder/${filename%.*}/%3d.png -y
 
 
done
