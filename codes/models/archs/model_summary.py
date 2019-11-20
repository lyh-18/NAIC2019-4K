from torchsummary import summary

from FastDVD_arch import FastDVDnet


model = FastDVDnet()


summary(model, (9, 64, 64))
