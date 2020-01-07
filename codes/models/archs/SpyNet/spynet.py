#!/usr/bin/env python

import torch
import math


def Backward(tensorInput, tensorFlow):
    Backward_tensorGrid = {}
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
            1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
            1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical],
                                                                1).cuda()

    tensorFlow = torch.cat([
        tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] /
        ((tensorInput.size(2) - 1.0) / 2.0)
    ], 1)

    return torch.nn.functional.grid_sample(
        input=tensorInput,
        grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1),
        mode='bilinear', padding_mode='border')


class SpyNet(torch.nn.Module):
    def __init__(self, N_levels=6, pretrain=True):
        super(SpyNet, self).__init__()

        self.N_levels = N_levels

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super(Preprocess, self).__init__()

            def forward(self, tensorInput):
                tensorRed = (tensorInput[:, 0:1, :, :] - 0.485) / 0.229
                tensorBlue = (tensorInput[:, 2:3, :, :] - 0.406) / 0.225
                tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224

                return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1,
                                    padding=3), torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1,
                                    padding=3), torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1,
                                    padding=3), torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1,
                                    padding=3), torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1,
                                    padding=3))

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)

        self.modulePreprocess = Preprocess()
        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(6)])
        if pretrain:
            self.load_state_dict(torch.load('models/modules/SpyNet/network-sintel-final.pytorch'))
        self.moduleBasic = self.moduleBasic[:N_levels]

    def process(self, ref, supp):
        flow = []

        ref = [self.modulePreprocess(ref)]
        supp = [self.modulePreprocess(supp)]

        for intLevel in range(self.N_levels - 1):
            # if ref[0].size(2) > 32 or ref[0].size(3) > 32:
            ref.insert(
                0,
                torch.nn.functional.avg_pool2d(input=ref[0], kernel_size=2, stride=2,
                                               count_include_pad=False))
            supp.insert(
                0,
                torch.nn.functional.avg_pool2d(input=supp[0], kernel_size=2, stride=2,
                                               count_include_pad=False))

        flow = ref[0].new_zeros([
            ref[0].size(0), 2,
            int(math.floor(ref[0].size(2) / 2.0)),
            int(math.floor(ref[0].size(3) / 2.0))
        ])

        for intLevel in range(len(ref)):
            upsampled = torch.nn.functional.interpolate(input=flow, scale_factor=2, mode='bilinear',
                                                        align_corners=True) * 2.0

            if upsampled.size(2) != ref[intLevel].size(2):
                upsampled = torch.nn.functional.pad(input=upsampled, pad=[0, 0, 0, 1],
                                                    mode='replicate')
            if upsampled.size(3) != ref[intLevel].size(3):
                upsampled = torch.nn.functional.pad(input=upsampled, pad=[0, 1, 0, 0],
                                                    mode='replicate')

            flow = self.moduleBasic[intLevel](torch.cat([
                ref[intLevel],
                Backward(tensorInput=supp[intLevel], tensorFlow=upsampled), upsampled
            ], 1)) + upsampled

        return flow

    def forward(self, ref, supp):
        assert (ref.size(2) == supp.size(2))
        assert (ref.size(3) == supp.size(3))

        H, W = ref.size(2), ref.size(3)

        W_floor = int(math.floor(math.ceil(W / 32.0) * 32.0))
        H_floor = int(math.floor(math.ceil(H / 32.0) * 32.0))

        ref = torch.nn.functional.interpolate(input=ref, size=(H_floor, W_floor), mode='bilinear',
                                              align_corners=False)
        supp = torch.nn.functional.interpolate(input=supp, size=(H_floor, W_floor), mode='bilinear',
                                               align_corners=False)

        flow = torch.nn.functional.interpolate(input=self.process(ref, supp), size=(H, W),
                                               mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(W) / float(W_floor)
        flow[:, 1, :, :] *= float(H) / float(H_floor)

        return flow
