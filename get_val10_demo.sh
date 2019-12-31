#!/bin/bash

mkdir /tmp/data/val1_HR_png
mkdir /tmp/data/val1_LR_png
mkdir /tmp/data/val1_HR_png
mkdir /tmp/data/val2_LR_png


VAL_LIST="11553969 17422120 19524575 21043777 30063783 30672625 40267971 47682563 56318324 70571139"
for i in $VAL_LIST
do
  mv /tmp/data/train1_HR_png/$i /tmp/data/val1_HR_png
  mv /tmp/data/train1_LR_png/$i /tmp/data/val1_LR_png
  mv /tmp/data/train2_LR_png/$i /tmp/data/val2_LR_png
done

