from sqlite3 import paramstyle
import sys
from turtle import back
sys.path.append('C:/my_github/PyTorch-Object-Detection')

import torch
from torch import nn
import torchsummary
import torchvision.models as models

from models.initialize import weight_initialize
from models.layers.conv_block import Conv2dBnRelu
from models.backbone.darknet import darknet19


class YoloV1(nn.Module):
    def __init__(self, backbone, backbone_out_features, num_classes, num_boxes):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.yolov1_head = nn.Sequential(
            Conv2dBnRelu(backbone_out_features, 1024, 3, 2),
            Conv2dBnRelu(1024, 1024, 3, 2),
            Conv2dBnRelu(1024, 1024, 3, 1),
            Conv2dBnRelu(1024, 1024, 3, 1),
            nn.Flatten(),
            nn.Linear(1024*4*4, 512),
            nn.Linear(512, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 7*7*(self.num_classes + (5*self.num_boxes)))
        )
        
        weight_initialize(self.yolov1_head)

    def forward(self, x):
        # backbone forward
        x = self.backbone(x)

        # prediction
        predictions = self.yolov1_head(x)

        return predictions.view(-1, 7, 7, (self.num_classes + (5*self.num_boxes)))


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
 

if __name__ == '__main__':
    backbone = models.vgg16(pretrained=True)
    backbone = nn.Sequential(*list(backbone.features.children()))
    set_parameter_requires_grad(backbone, True)

    model = YoloV1(
        backbone=backbone,
        backbone_out_features=512,
        num_classes=3,
        num_boxes=2
    )

    torchsummary.summary(model, (3, 448, 448), batch_size=1, device='cpu')
    
    # print(backbone.fc.in_features)
    # model = YoloV1(
    #     backbone=darknet19,
    #     num_classes=20,
    #     num_boxes=2
    # )
    # print(model(torch.rand(1, 3, 448, 448)).shape)
    # print(model)
    # torchsummary.summary(model, (3, 448, 448), batch_size=1, device='cpu')
    