import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary

from models.layers.conv_block import Conv2dBnRelu
from utils.module_select import get_model


class YoloV3(nn.Module):
    def __init__(self, backbone, num_classes, num_anchors):
        super().__init__()

        assert num_anchors == 9

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_anchors = int(num_anchors/3)

        self.c3_conv = nn.Sequential(
            Conv2dBnRelu(384, 128, 1),
            Conv2dBnRelu(128, 256, 3),
            Conv2dBnRelu(256, 128, 1),
        )

        self.c4_conv = nn.Sequential(
            Conv2dBnRelu(768, 256, 1),
            Conv2dBnRelu(256, 512, 3),
            Conv2dBnRelu(512, 256, 1),
        )

        self.c5_conv = nn.Sequential(
            Conv2dBnRelu(1024, 512, 1),
            Conv2dBnRelu(512, 1024, 3),
            Conv2dBnRelu(1024, 512, 1)
        )

        self.c4_route = nn.Sequential(
            Conv2dBnRelu(256, 128, 3),
            
            nn.Upsample(scale_factor=2)
        )

        self.c5_route = nn.Sequential(
            Conv2dBnRelu(512, 256, 3),

            nn.Upsample(scale_factor=2)
        )

        self.p3_head = nn.Sequential(
            Conv2dBnRelu(128, 256, 3),
            
            nn.Conv2d(256, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )

        self.p4_head = nn.Sequential(
            Conv2dBnRelu(256, 512, 3),
            
            nn.Conv2d(512, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )
        
        self.p5_head = nn.Sequential(
            Conv2dBnRelu(512, 1024, 3),
            
            nn.Conv2d(1024, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )

    def forward(self, x):
        # backbone forward
        x = self.backbone.stem(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2) # [batch_size, 256, input_size/8, input_size/8]
        c4 = self.backbone.layer4(c3) # [batch_size, 512, input_size/16, input_size/16]
        c5 = self.backbone.layer5(c4) # [batch_size, 1024, input_size/32, input_size/32]

        # Prediction Branch
        c5 = self.c5_conv(c5)
        p5 = self.p5_head(c5)

        # Prediction Branch
        c5_route = self.c5_route(c5)
        c4 = torch.cat((c5_route, c4), 1) # [batch_size, 768, input_size/16, input_size/16]
        c4 = self.c4_conv(c4)
        p4 = self.p4_head(c4)

        # Prediction Branch
        c4_route = self.c4_route(c4)
        c3 = torch.cat((c4_route, c3), 1) # [batch_size, 384, input_size/8, input_size/8]
        c3 = self.c3_conv(c3)
        p3 = self.p3_head(c3)

        return p3, p4, p5


if __name__ == '__main__':
    input_size = 416

    backbone = get_model('darknet19')()

    model = YoloV3(
        backbone=backbone,
        num_classes=20,
        num_anchors=9
    )

    tmp_input = torch.randn((1, 3, input_size, input_size))

    p3, p4, p5 = model(tmp_input)

    print(f'p3: {p3.size()}')
    print(f'p4: {p4.size()}')
    print(f'p5: {p5.size()}')

    torchsummary.summary(model, (3, input_size, input_size), batch_size=1, device='cpu')

    # from module.yolov3_detector import YoloV3Detector
    # from utils.yaml_helper import get_configs

    # model = YoloV3Detector(
    #     model=model,
    #     cfg=get_configs('configs/yolov3_voc.yaml')
    # )
    # file_path = 'model.onnx'
    # input_sample = torch.randn((1, 3, 416, 416))
    # model.to_onnx(file_path, input_sample, export_params=True)
