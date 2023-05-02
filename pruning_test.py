import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
from torchmetrics import Accuracy
import cv2
import os
import torchvision
import torchsummary
import time
import sys
from models.backbone.darknet import darknet19
from models.layers.conv_block import *

# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

model1 = darknet19()
model1.to(device=device)

tmp_input = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
for _ in range(5):
    output = model1(tmp_input)

taken_time = 0
for _ in range(10):
    start = time.time()
    output = model1(tmp_input)
    end = time.time()
    taken_time += (end-start)
print(f"Taken Time: {(taken_time/10)*1000:.4f}ms")

for module in model1.modules():
    # print(module)
    if isinstance(module, torch.nn.Conv2d):
        # print(module.weight)
        print(list(module.named_buffers()))
        break

for module in model1.modules():
    if isinstance(module, torch.nn.Conv2d):
        # prune.l1_unstructured(module, name="weight", amount=0.5)
        prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
        prune.remove(module, "weight")

for module in model1.modules():
    # print(module)
    if isinstance(module, torch.nn.Conv2d):
        # print(module.weight)
        print(list(module.named_buffers()))
        break

taken_time = 0
for _ in range(10):
    start = time.time()
    output = model1(tmp_input)
    end = time.time()
    taken_time += (end-start)
print(f"Taken Time: {(taken_time/10)*1000:.4f}ms")


print(model1.state_dict().keys())
print(type(model1.state_dict()['layer1.1.conv.weight']))
print(model1.state_dict()['layer1.1.conv.weight'].size())
print('\n\n')