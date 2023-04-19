import torch
from torch import nn
import numpy as np
from torchmetrics import Accuracy
import cv2
import os
from bisect import bisect_right, bisect_left
from torchvision.models import resnet50, resnet34, efficientnet_b0, vgg16
import torchsummary
from torchinfo import summary
from utils.yolov2_utils import bbox_iou
from torchvision.models import vgg16
import timm
import matplotlib.pyplot as plt
from collections import OrderedDict


'''
save weights only specific layers
'''
# from models.detector.yolov4_tiny import YoloV4TinyV4

# model = YoloV4TinyV4(20, 9)

# ckpt_path = os.path.join(os.getcwd(), 'saved/yolov4-tiny_coco-person/version_0/checkpoints/epoch=184-step=185369.ckpt')
# checkpoint = torch.load(ckpt_path)
# state_dict = checkpoint['state_dict']
# new_state_dict = OrderedDict()
# for key in list(state_dict):
#     layer_name = key.split('.')[1]
#     if layer_name == 'b4_conv':
#         break
#     new_state_dict[key.replace("model.", "")] = state_dict.pop(key)

# torch.save(new_state_dict, 'pretrained_weights.pt')
# new_state_dict = torch.load(os.path.join(os.getcwd(), 'saved/yolov4-tiny_coco-person/version_0/checkpoints/pretrained_weights.pt'))

# model.load_state_dict(new_state_dict, False)

# Check param values
# for name1, m1 in model.named_children():
#     print(name1, m1)
#     for name2, m2 in m1.named_children():
#         print(name2, m2)
#         for param in m2.parameters():
#             print(param[0, 0, 0, :])
#             break
#         break
#     break


'''
Convert lightning ckpt file to pytorch pt file
'''
ckpt_path = os.path.join(os.getcwd(), 'saved/yolov4-tiny_coco-person/version_0/checkpoints/epoch=184-step=185369.ckpt')
checkpoint = torch.load(ckpt_path)
state_dict = checkpoint['state_dict']

new_state_dict = OrderedDict()
for key in list(state_dict):
    new_state_dict[key.replace("model.", "")] = state_dict.pop(key)

torch.save(new_state_dict, 'pretrained_weights.pt')