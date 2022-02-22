from torch import nn
import torch
from models.initialize import weight_initialize

from models.layers.conv_block import Conv2dBnRelu


class OSAModule(nn.Module):
    def __init__(self, in_channels, conv_channels, layers_per_block, trans_ch):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.layers_per_block = layers_per_block
        self.trans_ch = trans_ch
        self.layers = []
        transition_in_ch = in_channels

        for i in range(layers_per_block):
            self.layers.append(Conv2dBnRelu(
                self.in_channels, self.conv_channels, 3))
            self.in_channels = self.conv_channels
            transition_in_ch += self.conv_channels
        self.layers = nn.ModuleList(self.layers)
        self.transition = Conv2dBnRelu(transition_in_ch, trans_ch, 1)

    def forward(self, x):
        outputs = []
        outputs.append(x)
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        x = torch.cat(outputs, dim=1)
        x = self.transition(x)
        return x


class _VoVNet19(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.stage_channels = []

        self.stem = Conv2dBnRelu(in_channels, 64, 3, 2)

        self.layer1 = Conv2dBnRelu(64, 64, 3, 1)
        self.stage_channels.append(64)

        self.in_channels = 128
        self.layer2 = nn.Sequential(
            Conv2dBnRelu(64, 128, 3, 2),
            OSAModule(
                in_channels=self.in_channels,
                conv_channels=64,
                layers_per_block=3,
                trans_ch=128))
        self.stage_channels.append(128)

        self.in_channels = 128
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, 2, 0),
            OSAModule(
                in_channels=self.in_channels,
                conv_channels=80,
                layers_per_block=3,
                trans_ch=256))
        self.stage_channels.append(256)

        self.in_channels = 256
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2, 2, 0),
            OSAModule(
                in_channels=self.in_channels,
                conv_channels=96,
                layers_per_block=3,
                trans_ch=384))
        self.stage_channels.append(384)

        self.in_channels = 384
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2, 2, 0),
            OSAModule(
                in_channels=self.in_channels,
                conv_channels=112,
                layers_per_block=3,
                trans_ch=512))
        self.stage_channels.append(512)

        self.in_channels = 512
        self.classification = nn.Sequential(
            Conv2dBnRelu(self.in_channels, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, x):
        y = self.stem(x)
        s1 = self.layer1(y)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)

        stages = [s1, s2, s3, s4, s5]
        for s in stages:
            print(s.shape)

        return {'stages': stages, 'pred': pred}


class _VoVNet27(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.stage_channels = []

        self.stem = Conv2dBnRelu(in_channels, 64, 3, 2)

        self.layer1 = Conv2dBnRelu(64, 64, 3, 1)
        self.stage_channels.append(64)

        self.in_channels = 128
        self.layer2 = nn.Sequential(
            Conv2dBnRelu(64, 128, 3, 2),
            OSAModule(
                in_channels=self.in_channels,
                conv_channels=64,
                layers_per_block=5,
                trans_ch=128))
        self.stage_channels.append(128)

        self.in_channels = 128
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, 2, 0),
            OSAModule(
                in_channels=self.in_channels,
                conv_channels=80,
                layers_per_block=5,
                trans_ch=256))
        self.stage_channels.append(256)

        self.in_channels = 256
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2, 2, 0),
            OSAModule(
                in_channels=self.in_channels,
                conv_channels=96,
                layers_per_block=5,
                trans_ch=384))
        self.stage_channels.append(384)

        self.in_channels = 384
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2, 2, 0),
            OSAModule(
                in_channels=self.in_channels,
                conv_channels=112,
                layers_per_block=5,
                trans_ch=512))
        self.stage_channels.append(512)

        self.in_channels = 512
        self.classification = nn.Sequential(
            Conv2dBnRelu(self.in_channels, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )

    def forward(self, x):
        y = self.stem(x)
        s1 = self.layer1(y)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)
        pred = self.classification(s5)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)

        stages = [s1, s2, s3, s4, s5]
        for s in stages:
            print(s.shape)

        return {'stages': stages, 'pred': pred}


def VoVNet27(in_channels, classes=1000):
    model = _VoVNet27(in_channels, classes)
    weight_initialize(model)
    return model


def VoVNet19(in_channels, classes=1000):
    model = _VoVNet27(in_channels, classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = VoVNet19(in_channels=3, classes=1000)
    print(model(torch.rand(1, 3, 320, 320)))
