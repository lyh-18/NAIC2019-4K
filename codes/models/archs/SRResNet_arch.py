import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as mutil


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class ResNet_alpha_beta_sconv(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(ResNet_alpha_beta_sconv, self).__init__()
        
  
        self.conv_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)                 # 64f       
        basic_block_64 = functools.partial(mutil.ResidualBlock_noBN, nf=nf)        
        self.encoder1 = mutil.make_layer(basic_block_64, 2)                    # 64f
        self.down_conv_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        
        self.conv_2 = nn.Conv2d(nf, nf*2, 3, 1, 1, bias=True)                  # 128f
        basic_block_128 = functools.partial(mutil.ResidualBlock_noBN, nf=nf*2) 
        self.encoder2 = mutil.make_layer(basic_block_128, 2)                   # 128f
        self.down_conv_2 = nn.Conv2d(nf*2, nf*2, 4, 2, 1, bias=False)
        
        self.conv_3 = nn.Conv2d(nf*2, nf*4, 3, 1, 1, bias=True)                # 256f
        basic_block_256 = functools.partial(mutil.ResidualBlock_noBN, nf=nf*4) 
        self.encoder3 = mutil.make_layer(basic_block_256, 2)                   # 256f
        self.down_conv_3 = nn.Conv2d(nf*4, nf*4, 4, 2, 1, bias=False)
        
        self.conv_4 = nn.Conv2d(nf*4, nf, 3, 1, 1, bias=True)    # 64f
        self.conv_5 = nn.Conv2d(nf, 6, 3, 1, 1, bias=True)       # 6f
        
        # pooling
        #self.avg_pool = nn.AvgPool2d(2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        

    def forward(self, x):
        x_in = x
        fea = self.lrelu(self.conv_1(x))
        fea = self.lrelu(self.encoder1(fea))
        fea = self.lrelu(self.down_conv_1(fea))
        #print('fea 1: ', fea.size())
        
        fea = self.lrelu(self.conv_2(fea))
        fea = self.lrelu(self.encoder2(fea))
        fea = self.lrelu(self.down_conv_2(fea))
        #print('fea 2: ', fea.size())
        
        fea = self.lrelu(self.conv_3(fea))
        fea = self.lrelu(self.encoder3(fea))
        fea = self.lrelu(self.down_conv_3(fea))
        #print('fea 3: ', fea.size())
        
        fea = self.lrelu(self.conv_4(fea))
        fea = self.conv_5(fea)
        fea = self.global_avg_pool(fea)
        alpha = fea[:,0:3,:,:]
        beta = fea[:,3:,:,:]
        #print('alpha: ', alpha.size())
        #print('beta: ', beta.size())
        
        out = alpha * x_in + beta
                
        return out

class ResNet_alpha_beta_sconv_3d(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(ResNet_alpha_beta_sconv_3d, self).__init__()
        
  
        self.conv_1 = nn.Conv3d(in_channels=in_nc, out_channels=nf, kernel_size=(3, 3, 3), stride=1, padding=(0,0,0), bias=True)
        basic_block_64 = functools.partial(mutil.ResidualBlock_noBN_3D, nf=nf)        
        self.encoder1 = mutil.make_layer(basic_block_64, 2)                    # 64f
        self.down_conv_1 = nn.Conv3d(nf, nf, (1, 4, 4), (1, 2, 2), (0, 1, 1), bias=False)
        
        self.conv_2 = nn.Conv3d(nf, nf*2, (3, 3, 3), stride=1, padding=(0,0,0), bias=True)                  # 128f
        basic_block_128 = functools.partial(mutil.ResidualBlock_noBN_3D, nf=nf*2) 
        self.encoder2 = mutil.make_layer(basic_block_128, 2)                   # 128f
        self.down_conv_2 = nn.Conv3d(nf*2, nf*2, (1, 4, 4), (1, 2, 2), (0, 1, 1), bias=False)
        
        self.conv_3 = nn.Conv3d(nf*2, nf*4, (3, 3, 3), stride=1, padding=(0,0,0), bias=True)                # 256f
        basic_block_256 = functools.partial(mutil.ResidualBlock_noBN_3D, nf=nf*4) 
        self.encoder3 = mutil.make_layer(basic_block_256, 2)                   # 256f
        self.down_conv_3 = nn.Conv3d(nf*4, nf*4, (1, 4, 4), (1, 2, 2), (0, 1, 1), bias=False)
        
        self.conv_4 = nn.Conv3d(nf*4, nf, (3,3,3), stride=1, padding=(0, 1, 1), bias=True)    # 64f
        self.conv_5 = nn.Conv3d(nf, 6, (3, 3, 3), stride=1, padding=(0, 1, 1), bias=True)       # 6f
        
        # pooling
        #self.avg_pool = nn.AvgPool2d(2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # padding
        self.pad_3d = nn.ReplicationPad3d(1)

        

    def forward(self, x):
        x_in = x
        fea = self.lrelu(self.conv_1(self.pad_3d(x)))      # [B, 64, 5, H, W]
        fea = self.lrelu(self.encoder1(fea))
        fea = self.lrelu(self.down_conv_1(fea))
        #print('fea 1: ', fea.size())
        
        fea = self.lrelu(self.conv_2(self.pad_3d(fea)))
        fea = self.lrelu(self.encoder2(fea))
        fea = self.lrelu(self.down_conv_2(fea))
        #print('fea 2: ', fea.size())
        
        fea = self.lrelu(self.conv_3(self.pad_3d(fea)))
        fea = self.lrelu(self.encoder3(fea))
        fea = self.lrelu(self.down_conv_3(fea))
        #print('fea 3: ', fea.size())
        
        fea = self.lrelu(self.conv_4(fea))
        fea = self.conv_5(fea)
        fea = fea.squeeze(2)
        fea = self.global_avg_pool(fea)
        alpha = fea[:,0:3,:,:]
        beta = fea[:,3:,:,:]
        #print('alpha: ', alpha.size())
        #print('beta: ', beta.size())
        
        out = alpha * x_in + beta
                
        return out

class ResNet_alpha_beta_decoder_3x3(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(ResNet_alpha_beta_decoder_3x3, self).__init__()
        
        # encoder
        self.conv_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)                 # 64f       
        basic_block_64 = functools.partial(mutil.ResidualBlock_noBN, nf=nf)        
        self.encoder1 = mutil.make_layer(basic_block_64, 2)                    # 64f
        
        self.conv_2 = nn.Conv2d(nf, nf*2, 3, 1, 1, bias=True)                  # 128f
        basic_block_128 = functools.partial(mutil.ResidualBlock_noBN, nf=nf*2) 
        self.encoder2 = mutil.make_layer(basic_block_128, 2)                   # 128f
        
        self.conv_3 = nn.Conv2d(nf*2, nf*4, 3, 1, 1, bias=True)                # 256f
        basic_block_256 = functools.partial(mutil.ResidualBlock_noBN, nf=nf*4) 
        self.encoder3 = mutil.make_layer(basic_block_256, 2)                   # 256f
        
        self.conv_4 = nn.Conv2d(nf*4, nf, 3, 1, 1, bias=True)    # 64f
        self.conv_5 = nn.Conv2d(nf, 6, 3, 1, 1, bias=True)       # 6f
        
        # pooling
        self.avg_pool = nn.AvgPool2d(2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        
        # decoder
        self.conv_6 = nn.Conv2d(6, 32, 3, 1, padding=1, bias=True)
        self.conv_7 = nn.Conv2d(96, 32, 3, 1, padding=1, bias=True)
        self.conv_8 = nn.Conv2d(32, 32, 3, 1, padding=1, bias=True)
        self.conv_9 = nn.Conv2d(288, 32, 3, 1, padding=1, bias=True)
        self.conv_10 = nn.Conv2d(32, 32, 3, 1, padding=1, bias=True)
        self.conv_11 = nn.Conv2d(160, 32, 3, 1, padding=1, bias=True)
        self.conv_12 = nn.Conv2d(32, 32, 3, 1, padding=1, bias=True)
        self.conv_13 = nn.Conv2d(96, 32, 3, 1, padding=1, bias=True)
        self.conv_14 = nn.Conv2d(32, 6, 3, 1, padding=1, bias=True)

        

    def forward(self, x):
        B, C, H, W = x.size()
        x_in = x
        
        # encoder
        fea = self.lrelu(self.conv_1(x))
        fea_cat1 = self.encoder1(fea)      # [B, 64, H, W]
        fea = self.avg_pool(fea_cat1)
        fea = self.lrelu(self.conv_2(fea)) 
        fea_cat2 = self.encoder2(fea)      # [B, 128, H/2, W/2]
        fea = self.avg_pool(fea_cat2)
        fea = self.lrelu(self.conv_3(fea))
        fea_cat3 = self.encoder3(fea)      # [B, 256, H/4, W/4]
        fea = self.avg_pool(fea_cat3)
        fea_cat4 = self.conv_4(fea)        # [B, 64, H/8, W/8]
        fea = self.lrelu(fea_cat4)
        fea = self.conv_5(fea)
        fea = self.global_avg_pool(fea)    # [B, 6, 1, 1]
        
        
        # decoder
        de_fea = self.conv_6(fea)
        de_fea = F.upsample(de_fea, size=(H//8, W//8), mode='bilinear')
        de_fea_cat1 = torch.cat([fea_cat4, de_fea], 1)    # [B, 96, H/8, W/8]
        de_fea = self.lrelu(self.conv_7(de_fea_cat1))     # [B, 32, H/8, W/8]
        de_fea = self.conv_8(de_fea)                      # [B, 32, H/8, W/8]
        de_fea = F.upsample(de_fea, size=(H//4, W//4), mode='bilinear')
        de_fea_cat2 = torch.cat([fea_cat3, de_fea], 1)    # [B, 288, H/4, W/4]       
        de_fea = self.lrelu(self.conv_9(de_fea_cat2))     # [B, 32, H/4, W/4]    
        de_fea = self.conv_10(de_fea)                     # [B, 32, H/4, W/4]
        de_fea = F.upsample(de_fea, size=(H//2, W//2), mode='bilinear')
        de_fea_cat3 = torch.cat([fea_cat2, de_fea], 1)    # [B, 160, H/2, W/2]          
        de_fea = self.lrelu(self.conv_11(de_fea_cat3))    # [B, 32, H/2, W/2]
        de_fea = self.conv_12(de_fea)                     # [B, 32, H/2, W/2]
        de_fea = F.upsample(de_fea, size=(H, W), mode='bilinear')
        de_fea_cat4 = torch.cat([fea_cat1, de_fea], 1)    # [B, 96, H, W]
        de_fea = self.lrelu(self.conv_13(de_fea_cat4))    # [B, 32, H, W]
        de_fea = self.conv_14(de_fea)                     # [B, 6, H, W]
        #print('de_fea: ', de_fea.size())
         
        alpha = de_fea[:,0:3,:,:]
        beta = de_fea[:,3:,:,:]
        
        #print('alpha: ', alpha.size(), 'beta: ', beta.size())
        
        out = alpha * x_in + beta
                
        return out

class ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW, self).__init__()
        
        # encoder
        self.conv_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)                 # 64f       
        basic_block_64 = functools.partial(mutil.ResidualBlock_IN, nf=nf)        
        self.encoder1 = mutil.make_layer(basic_block_64, 2)                    # 64f
        
        self.conv_2 = nn.Conv2d(nf, nf*2, 3, 1, 1, bias=True)                  # 128f
        basic_block_128 = functools.partial(mutil.ResidualBlock_IN, nf=nf*2) 
        self.encoder2 = mutil.make_layer(basic_block_128, 2)                   # 128f
        
        self.conv_3 = nn.Conv2d(nf*2, nf*4, 3, 1, 1, bias=True)                # 256f
        basic_block_256 = functools.partial(mutil.ResidualBlock_IN, nf=nf*4) 
        self.encoder3 = mutil.make_layer(basic_block_256, 2)                   # 256f
        
        self.conv_4 = nn.Conv2d(nf*4, nf, 3, 1, 1, bias=True)    # 64f
        self.bn_4 = nn.InstanceNorm2d(nf, affine=True)
        self.conv_5 = nn.Conv2d(nf, 64, 3, 1, 1, bias=True)       # 64f
        
        # pooling
        self.avg_pool = nn.AvgPool2d(2)
        

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        
        # decoder
        self.conv_6 = nn.Conv2d(64, 32, 3, 1, padding=1, bias=True)
        
        self.conv_7 = nn.Conv2d(96, 32, 3, 1, padding=1, bias=True)
        
        self.conv_8 = nn.Conv2d(32, 32, 3, 1, padding=1, bias=True)
       
        self.conv_9 = nn.Conv2d(288, 32, 3, 1, padding=1, bias=True)
        
        self.conv_10 = nn.Conv2d(32, 32, 3, 1, padding=1, bias=True)
       
        self.conv_11 = nn.Conv2d(160, 32, 3, 1, padding=1, bias=True)
       
        self.conv_12 = nn.Conv2d(32, 32, 3, 1, padding=1, bias=True)
        
        self.conv_13 = nn.Conv2d(96, 32, 3, 1, padding=1, bias=True)
      
        self.conv_14 = nn.Conv2d(32, 6, 3, 1, padding=1, bias=True)

        

    def forward(self, x):
        B, C, H, W = x.size()
        x_in = x
        #x_mean = torch.mean(x, dim=[2,3])   # [B, 3]
        #x_std = torch.std(x, dim=[2,3])   # [B, 3]
        #x_mean = x_mean.unsqueeze(2).unsqueeze(2)  # [B, 3, 1, 1]
        #x_std = x_std.unsqueeze(2).unsqueeze(2)    # [B, 3, 1, 1]
        
        # encoder
        fea = self.lrelu(self.conv_1(x))
        fea_cat1 = self.encoder1(fea)      # [B, 64, H, W]
        fea = self.avg_pool(fea_cat1)
        fea = self.lrelu(self.conv_2(fea)) 
        fea_cat2 = self.encoder2(fea)      # [B, 128, H/2, W/2]
        fea = self.avg_pool(fea_cat2)
        fea = self.lrelu(self.conv_3(fea))
        fea_cat3 = self.encoder3(fea)      # [B, 256, H/4, W/4]
        fea = self.avg_pool(fea_cat3)
        fea_cat4 = self.bn_4(self.conv_4(fea))        # [B, 64, H/8, W/8]
        fea = self.lrelu(fea_cat4)
        fea = self.conv_5(fea)             # [B, 64, H/8, W/8]
        
        
        
        # decoder
        de_fea = (self.conv_6(fea))                       # [B, 64, H/8, W/8]
        de_fea_cat1 = torch.cat([fea_cat4, de_fea], 1)    # [B, 96, H/8, W/8]
        de_fea = self.lrelu((self.conv_7(de_fea_cat1)))     # [B, 32, H/8, W/8]
        de_fea = (self.conv_8(de_fea))                      # [B, 32, H/8, W/8]
        de_fea = F.upsample(de_fea, size=(H//4, W//4), mode='bilinear')
        de_fea_cat2 = torch.cat([fea_cat3, de_fea], 1)    # [B, 288, H/4, W/4]       
        de_fea = self.lrelu((self.conv_9(de_fea_cat2)))     # [B, 32, H/4, W/4]    
        de_fea = (self.conv_10(de_fea))                     # [B, 32, H/4, W/4]
        de_fea = F.upsample(de_fea, size=(H//2, W//2), mode='bilinear')
        de_fea_cat3 = torch.cat([fea_cat2, de_fea], 1)    # [B, 160, H/2, W/2]          
        de_fea = self.lrelu((self.conv_11(de_fea_cat3)))    # [B, 32, H/2, W/2]
        de_fea = (self.conv_12(de_fea))                     # [B, 32, H/2, W/2]
        de_fea = F.upsample(de_fea, size=(H, W), mode='bilinear')
        de_fea_cat4 = torch.cat([fea_cat1, de_fea], 1)    # [B, 96, H, W]
        de_fea = self.lrelu((self.conv_13(de_fea_cat4)))    # [B, 32, H, W]
        de_fea = self.conv_14(de_fea)                     # [B, 6, H, W]
        #print('de_fea: ', de_fea.size())
         
        alpha = de_fea[:,0:3,:,:]
        beta = de_fea[:,3:,:,:]
        
        #print('alpha: ', alpha.size(), 'beta: ', beta.size())
        
        out = alpha * x_in + beta
                
        return out


class ResNet_alpha_beta_multi_in(nn.Module):
    def __init__(self, in_nc=3, nf=64, structure='ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW'):
        super(ResNet_alpha_beta_multi_in, self).__init__()
        
        #if structure == 'ResNet_alpha_beta_decoder_3x3':
        #    self.color_net = ResNet_alpha_beta_decoder_3x3()
        
        if structure == 'ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW':
            print('Loading color model: ',structure)
            self.color_net = ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW()
        
        else:
            raise NotImplementedError('Color model [{:s}] not recognized'.format(structure))

        

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        
        corrected_x = []
        for i in range(N):
            corrected_x.append(self.color_net(x[:,i,:,:,:]))
        corrected_x = torch.stack(corrected_x, dim=1)
                
        return corrected_x