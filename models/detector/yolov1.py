import sys
sys.path.append('C:/my_github/PyTorch-Object-Detection')

import torch
from torch import nn
import torchsummary

from models.initialize import weight_initialize
from models.layers.conv_block import Conv2dBnRelu


class YoloV1(nn.Module):
    def __init__(self, backbone, num_classes, num_boxes, in_channels=3):
        super().__init__()

        self.backbone = backbone(in_channels, include_top=False)
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.yolov1_head = nn.Sequential(
            Conv2dBnRelu(1024, 1024, 3, 2),
            nn.Flatten(),
            nn.Linear(1024*7*7, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 7*7*(self.num_classes + (5*self.num_boxes)))
        )
        
        weight_initialize(self.yolov1_head)

    def forward(self, x):
        # backbone forward
        x = self.backbone(x)

        # prediction
        predictions = self.yolov1_head(x)

        return predictions.view(-1, 7, 7, (self.num_classes + (5*self.num_boxes)))


if __name__ == '__main__':
    from models.backbone.darknet import darknet19
    model = YoloV1(
        backbone=darknet19,
        num_classes=20,
        num_boxes=2
    )
    print(model(torch.rand(1, 3, 448, 448)).shape)
    torchsummary.summary(model, (3, 448, 448), batch_size=1, device='cpu')
    