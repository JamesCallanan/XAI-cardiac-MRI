#!/usr/bin/env python
# adapted from https://github.com/microsoft/landcover/blob/master/training/pytorch/models/unet.py
import torch
import torch.nn as nn
import json
import os
import numpy as np


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN8s(nn.Module):
    def __init__(self, model_opts):
        super(FCN8s, self).__init__()
        self.opts = model_opts['fcn_opts']
        n_class = self.opts["n_classes"]
        n_channels = self.opts["n_input_channels"]

        # conv1
        self.conv1_1, self.relu1_1, self.conv1_2, self.relu1_2, self.pool1 = self.conv(n_channels, 64, padding_init=100)
        # conv2
        self.conv2_1, self.relu2_1, self.conv2_2, self.relu2_2, self.pool2 = self.conv(64, 128)
        # conv3
        self.conv3_1, self.relu3_1, self.conv3_2, self.relu3_2, self.conv3_3, self.relu3_3, self.pool3 = self.conv(128, 256, extra=True)      
        # conv4
        self.conv4_1, self.relu4_1, self.conv4_2, self.relu4_2, self.conv4_3, self.relu4_3, self.pool4 = self.conv(256, 512, extra=True)      
        # conv5
        self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2, self.conv5_3, self.relu5_3, self.pool5 = self.conv(512, 512, extra=True)      

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def conv(self, in_size, out_size, padding_init=1, extra=False):
        conv1 = nn.Conv2d(in_size, out_size, 3, padding=padding_init)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_size, out_size, 3, padding=1)
        relu2 = nn.ReLU(inplace=True)
        pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        if extra:
            conv3 = nn.Conv2d(out_size, out_size, 3, padding=1)
            relu3 = nn.ReLU(inplace=True)
            return conv1, relu1, conv2, relu2, conv3, relu3, pool

        return conv1, relu1, conv2, relu2, pool

    def _initialize_weights(self): # TODO: update initialization of the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h
