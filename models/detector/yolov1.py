import sys
import os
import math
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary

from models.initialize import weight_initialize
from utils.module_select import get_model


class YoloV1(nn.Module):
    def __init__(self, backbone, num_classes, num_boxes, in_channels, input_size):
        super().__init__()

        self.backbone = backbone(in_channels).features
        _, c, h, w = self.backbone(torch.randn((1, 3, input_size, input_size), dtype=torch.float32)).size()
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        self.yolov1_head = nn.Sequential(
            nn.Conv2d(c, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.Conv2d(1024, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.Conv2d(1024, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Flatten(),

            nn.Dropout(0.5),
            nn.Linear(256*math.ceil(h/2)*math.ceil(w/2), 7*7*(self.num_classes + (5*self.num_boxes)))
        )
        
        weight_initialize(self.yolov1_head)

    def forward(self, x):
        # backbone forward
        x = self.backbone(x)

        # prediction
        predictions = self.yolov1_head(x)

        return predictions.view(-1, 7, 7, (self.num_classes + (5*self.num_boxes)))


if __name__ == '__main__':
    input_size = 448
    in_channels = 3

    backbone = get_model('darknet19')

    model = YoloV1(
        backbone=backbone,
        num_classes=20,
        num_boxes=2,
        in_channels=in_channels,
        input_size=input_size
    )

    torchsummary.summary(model, (3, input_size, input_size), batch_size=1, device='cpu')

    print(model(torch.randn(1, 3, input_size, input_size)).size())
