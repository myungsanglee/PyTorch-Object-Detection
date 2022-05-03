from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from module.lr_scheduler import CosineAnnealingWarmUpRestarts
from torchvision.models import resnet18

from models.backbone.vgg import vgg16, vgg16_bn
from models.backbone.darknet import darknet19


def get_model(model_name):
    model_dict = {
        'vgg16': vgg16,
        'vgg16_bn': vgg16_bn,
        'darknet19': darknet19,
        'resnet18': resnet18
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


def get_scheduler(scheduler_name, optim, **kwargs):
    scheduler_dict = {
        'multi_step': MultiStepLR, 
        'cosine_annealing_warm_restarts': CosineAnnealingWarmRestarts,
        'cosine_annealing_warm_up_restarts': CosineAnnealingWarmUpRestarts
    }
    optimizer = scheduler_dict.get(scheduler_name)
    if optimizer:
        return optimizer(optim, **kwargs)
