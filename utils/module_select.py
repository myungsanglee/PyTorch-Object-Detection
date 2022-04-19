from models.backbone.vgg import vgg16, vgg16_bn
from models.backbone.darknet import darknet19
from torch import optim


def get_model(model_name):
    model_dict = {
        'vgg16': vgg16(3).features,
        'vgg16_bn': vgg16_bn(3).features,
        'darknet19': darknet19(3).features
    }
    return model_dict.get(model_name)


def get_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {
        'sgd': optim.SGD, 
        'adam': optim.Adam,
        'radam': optim.RAdam
    }
    optimizer = optim_dict.get(optimizer_name)
    if optimizer:
        return optimizer(params, **kwargs)
