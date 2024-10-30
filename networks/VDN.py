#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06
import torch
import torch.nn as nn
from .DnCNN import DnCNN
from .UNet import UNet
from .SubBlocks import conv3x3

def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net

class VDN(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN, self).__init__()
        head_layer = []
        head_layer.append(conv3x3(in_channels, wf, bias=True))
        for ii in range(1, 5):
            head_layer.append(conv3x3(wf, wf, bias=True))
            head_layer.append(nn.LeakyReLU(slope, inplace=True))
        self.head_layer = nn.Sequential(*head_layer)

        self.DNet = UNet(wf, in_channels, wf=wf, depth=dep_U, slope=slope)
        self.SNet = DnCNN(wf, in_channels, dep=dep_S, num_filters=64, slope=slope)

    def forward(self, x, mode='train'):
        x = self.head_layer(x)
        phi_U = self.DNet(x)
        phi_n = self.SNet(x)
        if mode.lower() == 'train':
            return phi_n, phi_U
        elif mode.lower() == 'test':
            return phi_n, phi_U


