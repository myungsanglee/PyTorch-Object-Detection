import os
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
from torchinfo import summary

from utils.module_select import get_model
from models.layers.conv_block import Conv2dBnRelu, V4TinyBlock


class YoloV4TinyV4(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()

        assert num_anchors == 9

        self.num_classes = num_classes
        self.num_anchors = int(num_anchors/3)

        self.stem = nn.Sequential(
            Conv2dBnRelu(3, 32, 3, 2)
        )
        
        self.layer1 = nn.Sequential(
            Conv2dBnRelu(32, 64, 3, 2),
            Conv2dBnRelu(64, 64, 3, 1)
        )
        
        self.tiny_block1 = V4TinyBlock(64, 32)
        
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv2dBnRelu(128, 128, 3, 1)
        )
        
        self.tiny_block2 = V4TinyBlock(128, 64)
        
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv2dBnRelu(256, 256, 3, 1)
        )
        
        self.tiny_block3 = V4TinyBlock(256, 128)
        
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv2dBnRelu(512, 512, 3, 1),
            Conv2dBnRelu(512, 256, 1, 1)
        )
        
        self.b4_conv = Conv2dBnRelu(384, 256, 3)

        self.b4_route = nn.Sequential(
            Conv2dBnRelu(256, 64, 1),
            
            nn.Upsample(scale_factor=2)
        )

        self.b5_route = nn.Sequential(
            Conv2dBnRelu(256, 128, 1),

            nn.Upsample(scale_factor=2)
        )

        self.p3_head = nn.Sequential(
            Conv2dBnRelu(192, 128, 3),
            
            nn.Conv2d(128, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )

        self.p4_head = nn.Conv2d(256, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        
        self.p5_head = nn.Sequential(
            Conv2dBnRelu(256, 512, 3),
            
            nn.Conv2d(512, (self.num_anchors*(self.num_classes + 5)), 1, 1, bias=False)
        )

    def forward(self, x):
        x = self.stem(x)
        y = self.layer1(x)
        x = self.tiny_block1(y)
        x = torch.cat((y, x), dim=1)
        
        y = self.layer2(x)
        b3 = self.tiny_block2(y)
        x = torch.cat((y, b3), dim=1)
        
        y = self.layer3(x)
        b4 = self.tiny_block3(y)
        x = torch.cat((y, b4), dim=1)
        
        b5 = self.layer4(x)
        
        # Prediction Branch
        p5 = self.p5_head(b5)

        # Prediction Branch
        b5_route = self.b5_route(b5)
        b4 = torch.cat((b5_route, b4), 1) # [batch_size, 384, input_size/16, input_size/16]
        b4 = self.b4_conv(b4)
        p4 = self.p4_head(b4)

        # Prediction Branch
        b4_route = self.b4_route(b4)
        b3 = torch.cat((b4_route, b3), 1) # [batch_size, 192, input_size/8, input_size/8]
        p3 = self.p3_head(b3)

        return p3, p4, p5


if __name__ == '__main__':
    input_size = 416
    tmp_input = torch.randn((1, 3, input_size, input_size))

    model = YoloV4TinyV4(
        num_classes=20,
        num_anchors=9
    )

    p3, p4, p5 = model(tmp_input)

    print(f'p3: {p3.size()}')
    print(f'p4: {p4.size()}')
    print(f'p5: {p5.size()}')

    summary(model, input_size=(1, 3, input_size, input_size), device='cpu')


    '''
    Check param values
    '''
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
    

    '''
    Convert to onnx
    '''
    # from module.yolov3_detector import YoloV3Detector
    # from utils.yaml_helper import get_configs

    # model = YoloV3Detector(
    #     model=model,
    #     cfg=get_configs('configs/yolov3_voc.yaml')
    # )
    
    # model = YoloV3Detector.load_from_checkpoint(
    #     checkpoint_path='saved/yolov4-tiny_voc/version_0/checkpoints/epoch=199-step=43999.ckpt',
    #     model=model,
    #     cfg=get_configs('configs/yolov4-tiny_voc.yaml')
    # )
    
    # file_path = 'model.onnx'
    # input_sample = torch.randn((1, 3, 416, 416))
    # model.to_onnx(file_path, input_sample, export_params=True, opset_version=9)
    