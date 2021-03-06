import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'sr':  # PSNR-oriented super resolution
        from .SR_model import SRModel as M
    elif model == 'srgan':  # GAN-based super resolution, SRGAN / ESRGAN
        from .SRGAN_model import SRGANModel as M
    # video restoration
    elif model == 'video_base':
        from .Video_base_model import VideoBaseModel as M
    elif model == 'video_recurr':
        from .Video_recurr_model import VideoRecurrModel as M
    elif model == 'my_video_base':
        from .my_Video_base_model import VideoBaseModel as M
    elif model == 'my_video_pre_enhance':
        from .my_Video_base_model import VideoBaseModel as M
    elif model == 'my_video_base_res':
        from .my_Video_base_res_model import VideoBaseModel as M
    elif model == 'color_video_base':
        from .Color_Video_base_model import VideoBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
