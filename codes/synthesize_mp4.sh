#!/bin/bash

folder="/tmp/data/answer_png"
files=$(ls $folder)
save_folder="/tmp/data/answer"

for filename in $files
do
 echo $folder/$filename
 echo $save_folder/${filename%.*}
 
 #mkdir -p $save_folder
 
 ffmpeg -r 24000/1001 -i $folder/$filename/%3d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 $save_folder/${filename}.mp4 -y
 
 
done
