import sys
import os
import math
sys.path.append(os.getcwd())

import torch
from torch import nn
from torchinfo import summary

from utils.module_select import get_model
from models.layers.conv_block import Conv2dBnRelu
# from models.initialize import weight_initialize


class YoloV1(nn.Module):
    def __init__(self, backbone_features_module, num_classes, num_boxes):
        super().__init__()

        self.backbone_features_module = backbone_features_module
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        self.yolov1_head = nn.Sequential(
            Conv2dBnRelu(1024, 1024, 3),
            
            Conv2dBnRelu(1024, 1024, 3, 2),
            
            Conv2dBnRelu(1024, 1024, 3),

            Conv2dBnRelu(1024, 1024, 3),
            
            Conv2dBnRelu(1024, 256, 3),
            
            nn.Flatten(),

            nn.Dropout(0.5),
            
            nn.Linear(256*7*7, 7*7*(self.num_classes + (5*self.num_boxes)))
        )
        
        # weight_initialize(self.yolov1_head)

    def forward(self, x):
        # backbone forward
        x = self.backbone_features_module(x)

        # prediction
        predictions = self.yolov1_head(x)

        return predictions
        # return predictions.view(-1, 7, 7, (self.num_classes + (5*self.num_boxes)))


if __name__ == '__main__':
    input_size = 448
    tmp_input = torch.randn((1, 3, input_size, input_size))

    backbone_features_module = get_model('darknet19')(pretrained='', features_only=True)
    
    model = YoloV1(
        backbone_features_module=backbone_features_module,
        num_classes=20,
        num_boxes=2
    )

    summary(model, input_size=(1, 3, input_size, input_size), device='cpu')