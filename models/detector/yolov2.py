import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary

from utils.module_select import get_model
from models.layers.conv_block import Conv2dBnRelu


class YoloV2(nn.Module):
    def __init__(self, backbone, num_classes, num_anchors):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.b4_layer = nn.Sequential(
            Conv2dBnRelu(512, 64, 1)
        )

        self.b5_layer = nn.Sequential(
            Conv2dBnRelu(1024, 1024, 3),
            Conv2dBnRelu(1024, 1024, 3)
        )
        
        self.yolov2_head = nn.Sequential(
            Conv2dBnRelu(1280, 1024, 3),
            
            nn.Conv2d(1024, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )

        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        # backbone forward
        x = self.backbone.stem(x)
        b1 = self.backbone.layer1(x)
        b2 = self.backbone.layer2(b1)
        b3 = self.backbone.layer3(b2)
        b4 = self.backbone.layer4(b3)
        b5 = self.backbone.layer5(b4)

        b4 = self.b4_layer(b4)
        bs, _, h, w = b4.size()
        b4 = b4.view(bs, -1, h//2, w//2)

        b5 = self.b5_layer(b5)

        x = torch.cat((b4, b5), 1)

        x = self.dropout(x)

        # prediction
        predictions = self.yolov2_head(x)

        return predictions


if __name__ == '__main__':
    input_size = 416

    backbone = get_model('darknet19')()

    model = YoloV2(
        backbone=backbone,
        num_classes=20,
        num_anchors=5
    )

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
