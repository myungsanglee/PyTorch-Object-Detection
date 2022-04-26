from distutils.errors import LibError
import sys
sys.path.append('C:/my_github/PyTorch-Object-Detection')

import torch
from torch import nn, sigmoid
import torchsummary
import torchvision.models as models

from models.initialize import weight_initialize
from utils.module_select import get_model


class YoloV2(nn.Module):
    def __init__(self, backbone, num_classes, num_anchors):
        super().__init__()

        self.backbone = backbone[:17]
        self.reconv = backbone[17:]
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.yolov2_head = nn.Sequential(
            nn.Conv2d(3072, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.Conv2d(1024, (self.num_anchors*(self.num_classes + 5)), 1, 1)
        )
        
        weight_initialize(self.yolov2_head)

    def forward(self, x):
        # backbone forward
        passthrough = self.backbone(x)
        bs, _, h, w = passthrough.size()
        x = self.reconv(passthrough)
        passthrough = passthrough.view(bs, -1, h//2, w//2)
        x = torch.cat((passthrough, x), 1)

        # prediction
        predictions = self.yolov2_head(x)

        return predictions

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    input_size = 416

    backbone = get_model('darknet19')

    model = YoloV2(
        backbone=backbone,
        num_classes=20,
        num_anchors=5
    )

    torchsummary.summary(model, (3, input_size, input_size), batch_size=1, device='cpu')
    