FROM ubuntu:16.04


RUN apt-get update \
    && apt-get install -y \
            python3.6  \
    && pip install -y \
            tensorboard \
            future \
            lmdb \
            opencv-python \
            pyyaml \
            matplotlib \
            tqdm \
    && pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && add-apt-repository ppa:savoury1/ffmpeg4 \
    && add-apt-repository ppa:savoury1/graphics \
    && add-apt-repository ppa:savoury1/multimedia \
    && apt-get update \
    && apt-get install ffmpeg
    
