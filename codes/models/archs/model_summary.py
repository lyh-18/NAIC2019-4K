from torchsummary import summary

from pfrb import PFRB
import torch

model = PFRB(in_frames=5, in_channels=64, n_features=64).cuda()
print(model)

input = torch.randn(1,5,64,32,32).cuda()
out = model(input)


summary(model, (5, 64, 32, 32))
