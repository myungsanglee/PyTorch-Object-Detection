import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary
from torchinfo import summary

from utils.module_select import get_model
from models.layers.conv_block import Conv2dBnRelu
# from models.initialize import weight_initialize


# class YoloV2(nn.Module):
#     def __init__(self, backbone, num_classes, num_anchors):
#         super().__init__()

#         self.backbone = backbone
#         self.num_classes = num_classes
#         self.num_anchors = num_anchors

#         self.b4_layer = nn.Sequential(
#             Conv2dBnRelu(512, 64, 1)
#         )

#         self.b5_layer = nn.Sequential(
#             Conv2dBnRelu(1024, 1024, 3),
#             Conv2dBnRelu(1024, 1024, 3)
#         )
        
#         self.yolov2_head = nn.Sequential(
#             Conv2dBnRelu(1280, 1024, 3),
#             nn.Conv2d(1024, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
#         )

#         self.dropout = nn.Dropout2d(0.5)

#         # weight_initialize(self.b4_layer)
#         # weight_initialize(self.b5_layer)
#         # weight_initialize(self.yolov2_head)

#     def forward(self, x):
#         # backbone forward
#         x = self.backbone.stem(x)
#         b1 = self.backbone.layer1(x)
#         b2 = self.backbone.layer2(b1)
#         b3 = self.backbone.layer3(b2)
#         b4 = self.backbone.layer4(b3)
#         b5 = self.backbone.layer5(b4)

#         b4 = self.b4_layer(b4)
#         bs, _, h, w = b4.size()
#         b4 = b4.view(bs, -1, h//2, w//2)

#         b5 = self.b5_layer(b5)

#         x = torch.cat((b4, b5), 1)

#         x = self.dropout(x)

#         # prediction
#         predictions = self.yolov2_head(x)

#         return predictions


class YoloV2(nn.Module):
    def __init__(self, backbone_module_list, num_classes, num_anchors):
        super().__init__()

        self.backbone_module_list = backbone_module_list
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

        # weight_initialize(self.b4_layer)
        # weight_initialize(self.b5_layer)
        # weight_initialize(self.yolov2_head)

    def forward(self, x):
        # backbone forward
        for idx, module in enumerate(self.backbone_module_list):
            x = module(x)
            if idx == 4:
                b4 = x
            elif idx == 5:
                b5 = x

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

    backbone = get_model('darknet19')(pretrained='tiny-imagenet')

    model = YoloV2(
        backbone_module_list=backbone.get_features_module_list(),
        num_classes=20,
        num_anchors=5
    )

    # for name, module in model.named_children():
    #     print(name)
    #     # print(module)
    #     for n, child in module.named_children():
    #         print(n)
    #         print(child)
    #         for param in child.parameters():
    #             print(param[10, 2, 2, :])
    #             print(param[-1, -1, -1, :])
    #             print(param.requires_grad)
    #             break
    #         break
    #     break
    # print('')
    
    # torchsummary.summary(model, (3, input_size, input_size), batch_size=1, device='cpu')
    summary(model, input_size=(1, 3, input_size, input_size))
    
    
    '''
    Convert to onnx
    '''
    # from module.yolov2_detector import YoloV2Detector
    # from utils.yaml_helper import get_configs

    # model = YoloV2Detector(
    #     model=model,
    #     cfg=get_configs('configs/yolov2_voc.yaml')
    # )
    # file_path = 'model.onnx'
    # input_sample = torch.randn((1, 3, 416, 416))
    # model.to_onnx(file_path, input_sample, export_params=True)
