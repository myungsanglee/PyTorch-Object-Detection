import torch
from torch import nn


class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, padding_mode='zeros'):
        super(Conv2dBnRelu, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, False,
                              padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return self.relu(y)


class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, padding_mode='zeros',
                 non_linearity=None):
        super(Conv2dBnAct, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, False,
                              padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        if non_linearity is None:
            non_linearity = nn.ReLU
        self.act = non_linearity()
 
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return self.act(y)


class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, padding_mode='zeros'):
        super(Conv2dBn, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, False,
                              padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.conv(x)
        return self.bn(y)


class V4TinyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2dBnRelu(in_channels, out_channels, 3, 1)
        self.conv2 = Conv2dBnRelu(out_channels, out_channels, 3, 1)
        self.conv3 = Conv2dBnRelu(out_channels*2, out_channels*2, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        x = self.conv2(y)
        x = torch.cat((x, y), dim=1)
        x = self.conv3(x)
        
        return x