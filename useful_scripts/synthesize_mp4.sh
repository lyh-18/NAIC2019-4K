#!/bin/bash

folder="AI4K_test_A01b_150000"
files=$(ls $folder)
save_folder="AI4K_test_A01b_150000_mp4"

for filename in $files
do
 echo $folder/$filename
 echo $save_folder/${filename%.*}
 
 mkdir -p $save_folder
 #ffmpeg -i $folder/$filename -r 30 $save_folder/${filename%.*}/%3d.png -y
 ffmpeg -r 24000/1001 -i $folder/$filename/%3d.png -vcodec libx265 -pix_fmt yuv422p -crf 12 $save_folder/${filename}.mp4
 #break
 
done
