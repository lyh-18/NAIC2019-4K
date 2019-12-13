import torch
import math
import torch.nn as nn
import torch.nn.functional as F



class PFRB_Block(nn.Module):
    def __init__(self, in_frames=5, in_channels=64, n_features=64):
        super(PFRB_Block, self).__init__()
        self.in_frames = in_frames
        
        self.C1 = nn.Sequential(nn.Conv2d(in_channels, n_features, kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(negative_slope=0.2))
        self.C2 = nn.Sequential(nn.Conv2d(in_channels*in_frames, n_features, kernel_size=1, stride=1, padding=0),
                                nn.LeakyReLU(negative_slope=0.2))
        self.C3 = nn.Sequential(nn.Conv2d(n_features*2, in_channels, kernel_size=3, stride=1, padding=1),
                                nn.LeakyReLU(negative_slope=0.2))
        

    def forward(self, x):
        # x : torch.Size([b, in_frames, in_channels, H, W])        [b, 5, 64, 64, 64]
        I1_list = []
        for i in range(self.in_frames):
            xi = x[:, i, :, :, :]
            I1_list.append(self.C1(xi))        
        I1 = torch.cat(I1_list, 1)  
        
        I2 = self.C2(I1)
        
        I3_list = [torch.cat([I2, I1_list[i]],1) for i in range(self.in_frames)]
        output = []
        for i in range(self.in_frames):
            output.append(self.C3(I3_list[i])+x[:, i, :, :, :])
        output = torch.stack(output, dim=1)        
        # out : torch.Size([b, in_frames, in_channels, H, W])        [b, 5, 64, 64, 64]
        
        return x

class PFRB(nn.Module):
    def __init__(self, in_frames=5, in_channels=64, n_features=64, num_blocks=20):
        super(PFRB, self).__init__()
        self.PFRB_Block = PFRB_Block(in_frames, in_channels, n_features)
        self.PFRB_forward = [self.PFRB_Block for i in range(num_blocks)]
        
        print(self.PFRB_forward)
        
    def forward(self, x):
        
        return x
        