from core.model.mobilenet import generate_mobilenet
from core.model.efficientnet import generate_efficientnet
from core.model.resnet import generate_resnet
from core.model.timm_model import generate_timm_model
# from core.model.repvgg import generate_repvgg
# from core.model.multi import generate_multi_model


name_to_model = {
    # 'mobilenet_v3_small': generate_mobilenet,
    # 'mobilenet_v3_large': generate_mobilenet,
    # 'efficientnet_b0': generate_efficientnet,
    # 'efficientnet_b1': generate_efficientnet,
    # 'efficientnet_b2': generate_efficientnet,
    # 'efficientnet_b3': generate_efficientnet,
    # 'efficientnet_b4': generate_efficientnet,
    # 'resnet18': generate_resnet,
    
    'resnet18': generate_timm_model,
    # 'tf_efficientnetv2_b0': generate_timm_model,
    # 'tf_efficientnet_b0_ns': generate_timm_model,
    'efficientnet_b0': generate_timm_model,
    'mobilenetv3_large_100': generate_timm_model,
    # 'mobilenetv3_large_100_miil': generate_timm_model,
    'ig_resnext101_32x48d': generate_timm_model, 
    'swin_large_patch4_window7_224': generate_timm_model, 
    # 'repvgg-a0': generate_repvgg,
    # 'multi-model': generate_multi_model,
}



def get_model(args):
    return name_to_model[args.model](args)