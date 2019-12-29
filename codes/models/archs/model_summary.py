from torchsummary import summary

from pfrb import PFRB
from DynamicF import *
import torch

#model = PFRB(in_frames=5, in_channels=64, n_features=64).cuda()
#print(model)

#input = torch.randn(1,5,64,32,32).cuda()
#out = model(input)

'''
model = DFN_16L(scale=1, adapt_official=False).cuda()
#print(model)

input = torch.randn(1,7,3,64,64).cuda()
out = model(input)
print(out.size())
'''

model = DFN_16L_2d().cuda()
summary(model, (3, 32, 32))



#summary(model, (5, 64, 32, 32))
