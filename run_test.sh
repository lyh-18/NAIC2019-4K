#!/bin/bash


# specify folders
input_mp4_folder="/tmp/data/test"      
save_png_folder="/tmp/data/testA_LR_png"
output_png_folder="/tmp/data/answer_png"
output_mp4_folder="/tmp/data/answer"



# extract LR frames
files=$(ls $input_mp4_folder)
for filename in $files
do
 #echo $folder/$filename
 #echo $save_folder/${filename%.*}
 
 mkdir -p $save_png_folder/${filename%.*}
 ffmpeg -i $input_mp4_folder/$filename -r 24000/1001 $save_png_folder/${filename%.*}/%3d.png -y
 
 
done


# predict HR frames
INPUT_DIR=$save_png_folder

cd codes
python test_video_no_GT_color_demo.py \
  --input_folder="${save_png_folder}" \
  --save_folder="${output_png_folder}" \


# synthesize HR mp4
folder=$output_png_folder
files=$(ls $folder)
save_folder=$output_mp4_folder

for filename in $files
do
 echo $folder/$filename
 echo $save_folder/${filename%.*}
 
 #mkdir -p $save_folder
 
 ffmpeg -r 24000/1001 -i $folder/$filename/%3d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 $save_folder/${filename}.mp4 -y
 
 
done
