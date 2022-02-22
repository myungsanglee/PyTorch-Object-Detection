from models.backbone.vovnet import VoVNet19, VoVNet27
from models.backbone.vgg import vgg16, vgg16_bn
from models.detector.retinanet import ClassificationModel, RegressionModel
from models.detector.fpn import FeaturesPyramidNetwork
from torch import optim
from models.backbone.frostnet import FrostNet
from module.sam_optimizer import SAM


def get_model(model_name):
    model_dict = {'FrostNet': FrostNet,
                  'VoVNet19': VoVNet19, 
                  'VoVNet27': VoVNet27,
                  'vgg16': vgg16,
                  'vgg16_bn': vgg16_bn}
    return model_dict.get(model_name)


def get_fpn(fpn_name):
    fpn_dict = {'default': FeaturesPyramidNetwork}
    return fpn_dict.get(fpn_name)


def get_cls_subnet(subnet_name):
    subnet_dict = {'default': ClassificationModel}
    return subnet_dict.get(subnet_name)


def get_reg_subnet(subnet_name):
    subnet_dict = {'default': RegressionModel}
    return subnet_dict.get(subnet_name)


def get_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    optimizer = optim_dict.get(optimizer_name)
    print(f'\n[get_optimizer]\n{kwargs}\n')
    if optimizer:
        return optimizer(params, **kwargs)
    elif optimizer_name == 'sam':
        return SAM(params, optim.SGD, **kwargs)
