import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as mutil
import models.archs.SpyNet.spynet as spynet


class Conv_ResBlocks(nn.Module):
    def __init__(self, nf=64, N_RBs=15, in_dim=3, depthwise_separable=False):
        super(Conv_ResBlocks, self).__init__()
        if depthwise_separable:
            ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN_depthwise_separable,
                                                     nf=nf)
        else:
            ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)

        #### extract features
        self.conv_first = nn.Conv2d(in_dim, nf, 3, 1, 1, bias=True)
        self.res_blocks = mutil.make_layer(ResidualBlock_noBN_f, N_RBs)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        fea = self.res_blocks(fea)
        return fea


class Recurr_ResBlocks(nn.Module):
    def __init__(self, nf=64, N_RBs=15, N_flow_lv=6, pretrain_flow=True):
        super(Recurr_ResBlocks, self).__init__()

        self.nf = nf
        self.N_flow_lv = N_flow_lv

        if self.N_flow_lv > 0:
            self.flow = spynet.SpyNet(N_levels=N_flow_lv, pretrain=pretrain_flow)

        #### backward propagation
        self.att1_b = nn.Conv2d(nf + 3, nf, 3, 1, 1, bias=True)
        self.att2_b = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.att3_b = nn.Conv2d(nf, nf + 3, 3, 1, 1, bias=True)
        self.backward_ = Conv_ResBlocks(nf=nf, N_RBs=N_RBs, in_dim=nf + 3,
                                        depthwise_separable=False)

        #### forward propagation
        self.att1_f = nn.Conv2d(2 * nf + 3, nf, 3, 1, 1, bias=True)
        self.att2_f = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.att3_f = nn.Conv2d(nf, 2 * nf + 3, 3, 1, 1, bias=True)
        self.forward_ = Conv_ResBlocks(nf=nf, N_RBs=N_RBs, in_dim=2 * nf + 3,
                                       depthwise_separable=False)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        Given N LR images, recurrently 4x upsample all N images.
            input:
                x - (B, N, 3, H, W)
            output:
                out - (B, N, 3, 4H, 4W)
        '''
        N = x.size(1)  # Number of frames to reconstruct
        fea_forward = torch.zeros_like(x[:, 0, :1, :, :]).repeat(1, self.nf, 1, 1)
        fea_backward = fea_forward.clone()

        # backward propagation
        fea_backward_l = []
        for i in range(N - 1, -1, -1):
            x_current = x[:, i, :, :, :]

            if i < N - 1 and self.N_flow_lv > 0:
                flow = self.flow(x_current, x[:, i + 1, :, :, :])
                fea_backward = mutil.flow_warp(fea_backward, flow.permute(0, 2, 3, 1))
            else:
                flow = torch.zeros_like(x_current[:, :1, :, :]).repeat(1, 2, 1, 1)

            fea_backward = torch.cat([x_current, fea_backward], dim=1)
            att = self.lrelu(self.att1_b(fea_backward))
            att = self.lrelu(self.att2_b(att))
            att = self.sigmoid(self.att3_b(att))
            fea_backward = fea_backward * att
            fea_backward = self.backward_(fea_backward)

            fea_backward_l.insert(0, fea_backward)

        # forward propagation and reconstruction
        out_l = []
        for i in range(0, N):
            x_current = x[:, i, :, :, :]

            if i > 0 and self.N_flow_lv > 0:
                flow = self.flow(x_current, x[:, i - 1, :, :, :])
                fea_forward = mutil.flow_warp(fea_forward, flow.permute(0, 2, 3, 1))
            else:
                flow = torch.zeros_like(x_current[:, :1, :, :]).repeat(1, 2, 1, 1)

            fea_forward = torch.cat([x_current, fea_backward_l[i], fea_forward], dim=1)
            att = self.lrelu(self.att1_f(fea_forward))
            att = self.lrelu(self.att2_f(att))
            att = self.sigmoid(self.att3_f(att))
            fea_forward = fea_forward * att
            fea_forward = self.forward_(fea_forward)  # compute the features for update

            # reconstruction
            out = self.lrelu(self.pixel_shuffle(self.upconv1(fea_forward)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.HRconv(out))
            out = self.conv_last(out)
            base = F.interpolate(x_current, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l.append(out)

        out = torch.stack(out_l, dim=1)
        return out
