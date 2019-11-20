from torchsummary import summary

from models.archs.my_EDVR_arch import *



model = EDVR(nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10)
summary(model, (5, 3, 64, 64))