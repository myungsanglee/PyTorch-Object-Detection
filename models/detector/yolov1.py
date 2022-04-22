from distutils.errors import LibError
import sys
sys.path.append('C:/my_github/PyTorch-Object-Detection')

import torch
from torch import nn
import torchsummary
import torchvision.models as models

from models.initialize import weight_initialize
from utils.module_select import get_model


class YoloV1(nn.Module):
    def __init__(self, backbone, backbone_out_channels, num_classes, num_boxes):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        
        # version_0
        # self.yolov1_head = nn.Sequential(
        #     nn.Conv2d(backbone_out_channels, 1024, 3, 2, 1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
            
        #     nn.Conv2d(1024, 1024, 3, 2, 1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
            
        #     nn.Flatten(),
            
        #     nn.Linear(1024*4*4, 496),
        #     nn.ReLU(),
        #     nn.Linear(496, 7*7*(self.num_classes + (5*self.num_boxes)))
        # )
        
        # version_1
        self.yolov1_head = nn.Sequential(
            nn.Conv2d(backbone_out_channels, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.Conv2d(1024, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(1024*4*4, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 7*7*(self.num_classes + (5*self.num_boxes)))
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
    # vgg16 = models.vgg16(pretrained=True)

    # # backbone = nn.Sequential(*list(backbone.features.children()))
    # backbone = vgg16.features
    # torchsummary.summary(backbone, (3, 448, 448), batch_size=1, device='cpu')
    # set_parameter_requires_grad(backbone, True)

    backbone = get_model('darknet19')

    print(backbone(torch.randn((1, 3, 448, 448), dtype=torch.float32)).size())

    model = YoloV1(
        backbone=backbone,
        backbone_out_channels=backbone(torch.randn((1, 3, 448, 448), dtype=torch.float32)).size()[1],
        num_classes=3,
        num_boxes=2
    )

    torchsummary.summary(model, (3, 448, 448), batch_size=1, device='cpu')
