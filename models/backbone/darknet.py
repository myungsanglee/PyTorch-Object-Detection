import sys
sys.path.append('C:/my_github/PyTorch-Object-Detection')

import torch
from torch import nn
import torchsummary

from models.layers.conv_block import Conv2dBnRelu
from models.initialize import weight_initialize


class _Darknet19(nn.Module):
    def __init__(self, in_channels, num_classes, include_top=True):
        super(_Darknet19, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.include_top = include_top

        self.features = nn.Sequential(
            Conv2dBnRelu(in_channels, 32, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(32, 64, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(64, 128, 3),
            Conv2dBnRelu(128, 64, 1),
            Conv2dBnRelu(64, 128, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(128, 256, 3),
            Conv2dBnRelu(256, 128, 1),
            Conv2dBnRelu(128, 256, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(256, 512, 3),
            Conv2dBnRelu(512, 256, 1),
            Conv2dBnRelu(256, 512, 3),
            Conv2dBnRelu(512, 256, 1),
            Conv2dBnRelu(256, 512, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(512, 1024, 3),
            Conv2dBnRelu(1024, 512, 1),
            Conv2dBnRelu(512, 1024, 3),
            Conv2dBnRelu(1024, 512, 1),
            Conv2dBnRelu(512, 1024, 3)
        )
        
        self.classifier = nn.Sequential(
            Conv2dBnRelu(1024, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, num_classes, 1)
        )

    def forward(self, x):
        x = self.features(x)
        
        if self.include_top:
            pred = self.classifier(x)
            b, c, _, _ = pred.size()
            pred = pred.view(b, c)

            return {'pred': pred}

        else:
            return x


def darknet19(in_channels, num_classes=1000, include_top=True):
    model = _Darknet19(in_channels, num_classes, include_top)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = darknet19(in_channels=3, num_classes=200, include_top=True)
    # print(model(torch.rand(1, 3, 224, 224)).shape)
    # print(model)
    torchsummary.summary(model, (3, 448, 448), batch_size=1, device='cpu')
    
    # print(list(model.children()))
    # print(f'\n-------------------------------------------------------------\n')
    # new_model = nn.Sequential(*list(model.children())[:-1])
    # print(new_model.modules)
    
    # for idx, child in enumerate(model.children()):
    #     print(child)
    #     if idx == 0:
    #         for i, param in enumerate(child.parameters()):
    #             print(i, param)
    #             param.requires_grad = False
    #             if i == 4:
    #                 break

    # torchsummary.summary(model, (3, 64, 64), batch_size=1, device='cpu')