import torch
import logging
import models.modules.SRResNet_arch as SRResNet_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.DynamicF as DynamicF
logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    
    elif which_model == 'MResNet':
        netG = SRResNet_arch.MResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'ResNet_alpha_beta':
        netG = SRResNet_arch.ResNet_alpha_beta()
    elif which_model == 'ResNet_alpha_beta_sconv':
        netG = SRResNet_arch.ResNet_alpha_beta_sconv()
    elif which_model == 'ResNet_alpha_beta_fc':
        netG = SRResNet_arch.ResNet_alpha_beta_fc()
    elif which_model == 'ResNet_alpha_beta_fc_statistics':
        netG = SRResNet_arch.ResNet_alpha_beta_fc_statistics()
    elif which_model == 'ResNet_alpha_beta_decoder_1x1':
        netG = SRResNet_arch.ResNet_alpha_beta_decoder_1x1()
    elif which_model == 'ResNet_alpha_beta_decoder_3x3':
        netG = SRResNet_arch.ResNet_alpha_beta_decoder_3x3()
    elif which_model == 'ResNet_alpha_beta_decoder_3x3_BN':
        netG = SRResNet_arch.ResNet_alpha_beta_decoder_3x3_BN()
    elif which_model == 'ResNet_alpha_beta_decoder_3x3_IN':
        netG = SRResNet_arch.ResNet_alpha_beta_decoder_3x3_IN()
    elif which_model == 'ResNet_alpha_beta_decoder_3x3_IN_encoder':
        netG = SRResNet_arch.ResNet_alpha_beta_decoder_3x3_IN_encoder()
    elif which_model == 'ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW':
        netG = SRResNet_arch.ResNet_alpha_beta_decoder_3x3_IN_encoder_8HW()
    elif which_model == 'ResNet_alpha_beta_decoder_3x3_IN_encoder_global2local':
        netG = SRResNet_arch.ResNet_alpha_beta_decoder_3x3_IN_encoder_global2local()
    elif which_model == 'ResNet_plain':
        netG = SRResNet_arch.ResNet_plain()
                                    
    elif which_model == 'DFN':
        netG = DynamicF.DFN_16L_2d()
    elif which_model == 'DFN_1x1':
        netG = DynamicF.DFN_16L_2d_1x1()
    elif which_model == 'DFN_noRx':
        netG = DynamicF.DFN_16L_2d(res=False)
    elif which_model == 'DFN_alpha':
        netG = DynamicF.DFN_16L_2d_alpha()
    elif which_model == 'DCCN':
        netG = DynamicF.DCCN_16L_2d()
    elif which_model == 'DCCN_alpha':
        netG = DynamicF.DCCN_16L_2d_alpha()
                                    
    # elif which_model == 'sft_arch':  # SFT-GAN
    #     netG = sft_arch.SFT_Net()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    if which_model == 'discriminator_vgg_512':
        netD = SRGAN_arch.Discriminator_VGG_512(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
