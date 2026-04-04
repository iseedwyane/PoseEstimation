# MIT License.
# Copyright (c) 2022 by BioicDL. All rights reserved.
# Created by LiuXb on 2022/5/11
# -*- coding:utf-8 -*-

"""
@Modified: 
@Description: from imgs to 6d pose
"""
from turtle import forward
import torch
import torchvision
from torch import nn
import numpy as np

def conv3x3(in_channel, out_channel, stride=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    kernel_size = np.asarray((3,3))
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size-1)*(dilation-1) + kernel_size
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size-1)//2
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


class BasicBlock(nn.Module):
    """ basic resNet unit"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes*self.expansion, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes*self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


if __name__ == "__main__":
    test_input = torch.randn(1, 1, 7, 7)
    print(test_input)
    print(test_input.view(-1))

    print(test_input.shape)
    x = conv3x3(1, 1)
    k = x(test_input)
    print(k.shape)
    kk = BasicBlock(3, 5)
    print(kk)