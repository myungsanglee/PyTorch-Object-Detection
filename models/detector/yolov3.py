import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import dropout, nn
import torchsummary
import timm

from models.initialize import weight_initialize
from models.layers.conv_block import Conv2dBnRelu
from utils.module_select import get_model


class YoloV3(nn.Module):
    def __init__(self, backbone, num_classes, num_anchors):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_anchors = int(num_anchors/3)

        self.p4_route = nn.Sequential(
            Conv2dBnRelu(512, 256, 3),
            
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.p5_route = nn.Sequential(
            Conv2dBnRelu(1024, 512, 3),

            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.p3_head = nn.Sequential(
            Conv2dBnRelu(512, 256, 3),
            Conv2dBnRelu(256, 512, 3),
            
            nn.Conv2d(512, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )

        self.p4_head = nn.Sequential(
            Conv2dBnRelu(1024, 512, 3),
            Conv2dBnRelu(512, 1024, 3),
            
            nn.Conv2d(1024, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )
        
        self.p5_head = nn.Sequential(
            Conv2dBnRelu(1024, 512, 3),
            Conv2dBnRelu(512, 1024, 3),
            
            nn.Conv2d(1024, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )

        # self.dropout = nn.Dropout2d(p=0.5)


    def forward(self, x):
        # backbone forward
        x = self.backbone.stem(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2) # [batch_size, 256, input_size/8, input_size/8]
        c4 = self.backbone.layer4(c3) # [batch_size, 512, input_size/16, input_size/16]
        c5 = self.backbone.layer5(c4) # [batch_size, 1024, input_size/32, input_size/32]

        # Prediction Branch
        # c5 = self.dropout(c5)
        p5 = self.p5_head(c5)

        # Prediction Branch
        p5_route = self.p5_route(c5)
        p4 = torch.cat((p5_route, c4), 1)
        # p4 = self.dropout(p4)
        p4 = self.p4_head(p4)

        # Prediction Branch
        p4_route = self.p4_route(c4)
        p3 = torch.cat((p4_route, c3), 1)
        # p3 = self.dropout(p3)
        p3 = self.p3_head(p3)

        return p3, p4, p5


if __name__ == '__main__':
    input_size = 416

    backbone = get_model('darknet19')()

    model = YoloV3(
        backbone=backbone,
        num_classes=20,
        num_anchors=3
    )

    tmp_input = torch.randn((1, 3, input_size, input_size))

    p3, p4, p5 = model(tmp_input)

    print(f'p3: {p3.size()}')
    print(f'p4: {p4.size()}')
    print(f'p5: {p5.size()}')

    torchsummary.summary(model, (3, input_size, input_size), batch_size=1, device='cpu')

    # from module.yolov2_detector import YoloV2Detector
    # from utils.yaml_helper import get_configs

    # model = YoloV2Detector(
    #     model=model,
    #     cfg=get_configs('configs/yolov2_voc.yaml')
    # )
    # file_path = 'model.onnx'
    # input_sample = torch.randn((1, 3, 416, 416))
    # model.to_onnx(file_path, input_sample, export_params=True)
