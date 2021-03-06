import torch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.RRDBNet_arch as RRDBNet_arch
import models.archs.EDVR_arch as EDVR_arch
import models.archs.my_EDVR_arch as my_EDVR_arch
import models.archs.Recurr_arch as Recurr_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    # video restoration
    elif which_model == 'EDVR':
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'])
    elif which_model == 'MY_EDVR_FusionDenoise':
        netG = my_EDVR_arch.MYEDVR_FusionDenoise(
            nf=opt_net['nf'], nframes=opt_net['nframes'], groups=opt_net['groups'],
            front_RBs=opt_net['front_RBs'], back_RBs=opt_net['back_RBs'], center=opt_net['center'],
            predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'], w_TSA=opt_net['w_TSA'])
    elif which_model == 'MY_EDVR_RES':
        netG = my_EDVR_arch.MYEDVR_RES(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                       groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                       back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                       predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                                       w_TSA=opt_net['w_TSA'])
    elif which_model == 'MY_EDVR_PreEnhance':
        netG = my_EDVR_arch.MYEDVR_PreEnhance(
            nf=opt_net['nf'], nframes=opt_net['nframes'], groups=opt_net['groups'],
            front_RBs=opt_net['front_RBs'], back_RBs=opt_net['back_RBs'], center=opt_net['center'],
            predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'], w_TSA=opt_net['w_TSA'])

    elif which_model == 'Recurr_ResBlocks':
        netG = Recurr_arch.Recurr_ResBlocks(nf=opt_net['nf'], N_RBs=opt_net['N_RBs'],
                                            N_flow_lv=opt_net['N_flow_lv'],
                                            pretrain_flow=opt_net['pretrain_flow'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# pre color correction
def define_C(opt):
    opt_net = opt['network_C']
    which_model = opt_net['which_model_C']

    if which_model == 'dfn':
        netC = DynamicF.DFN_Color_correction()
    elif 'ResNet' in which_model:
        netC = SRResNet_arch.ResNet_alpha_beta_multi_in(which_model)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netC


# Define network used for perceptual loss
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
