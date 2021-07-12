#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/17 下午3:20
# @Author : PH
# @Version：V 0.1
# @File : custom_model.py
# @desc : Unet
import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
from torchsummary import summary


class TowLayerConv(nn.Module):
    """两层卷积"""

    def __init__(self, in_channels, out_channels, pad=1, dropout=0.1):
        """

        Args:
            in_channels: int
                输入图片的通道数
            out_channels: int
                卷积层输出的通道数
            pad: 0 or 1
                0表示不使用padding，即论文中的valid_pad,1表示使用大小为1的padding，即same_pad
            dropout: default to 0.1
        Attributes:
            self.conv1/2 : 使用BN时，卷积层有没有bias经过BN后都是一样的输出，因此就不要bias。减少参数量
        """
        super(TowLayerConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(pad, pad),
                               bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(pad, pad),
                               bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.norm1(self.relu(self.conv1(x)))
        x = self.norm2(self.relu(self.conv2(x)))
        x = self.dropout(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, features=[64, 128, 256, 512], same_pad=True):
        """

        Args:
            in_channels: default to 3
            n_classes: default to 1
                分割任务的类别数，为1时表示二分类问题
            features: default to [64, 128, 256, 512]
                Unet下采样和上采样时，每层卷积输出对应的通道数
            same_pad: default to False
                默认使用same卷积，注意当输入的原始图片的大小不是len(features)的整数倍时，
                跳层连接时，两个层的数据形状会不一样，因此需要resize一方，令其和另一方一样大小。
                因此虽说是same卷积。最后Unet的输出形状也可能不与原始输入一样。
                使用Valid卷积时，也会使用resize1的方法
        Attributes：
            self.downs: nn.ModuleList()
                卷积后下采样，和self.pooling一起使用
            self.ups: nn.ModuleList()
                上采样，其中带了反卷积
            self.bottleneck：TowLayerConv
                Unet下采样后与上采样中间的连接层
            self.out_layer： Conv2d
                使用1x1卷积来改变输出的通道数，令其和n_classes一样

        """
        super(Unet, self).__init__()
        pad = 1 if same_pad else 0
        self.n_classes = n_classes
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        # downs
        for feature in features:
            self.downs.append(TowLayerConv(in_channels, feature, pad))
            in_channels = feature

        self.bottleneck = TowLayerConv(features[-1], features[-1] * 2, pad)

        # ups
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=(2, 2), stride=(2, 2))
            )
            self.ups.append(TowLayerConv(feature * 2, feature, same_pad))

        self.out_layer = nn.Conv2d(features[0], n_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        skip_connects = []
        # down sample
        for down in self.downs:
            x = down(x)
            skip_connects.append(x)
            x = self.pooling(x)
        skip_connects = skip_connects[::-1]
        # U的中间层
        x = self.bottleneck(x)
        # up sample
        for i in range(0, len(self.ups), 2):
            # 反卷积
            x = self.ups[i](x)
            # 如果对应的up和down，跳层连接的h、w大小不一样，就要resize一方
            skip_connect = skip_connects[i // 2]
            if x.size(-1) != skip_connect.size(-1):
                skip_connect = TF.resize(skip_connect, size=x.shape[2:])
            # 跳层连接
            x = torch.cat((skip_connect, x), dim=1)
            # 两层卷积
            x = self.ups[i + 1](x)
        out = self.out_layer(x)
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inp = torch.rand(16, 3, 256, 256).to(device)
    unet = Unet(same_pad=True).to(device)
    print(unet(inp).shape)
    summary(unet, (3, 256, 256), batch_size=5)
