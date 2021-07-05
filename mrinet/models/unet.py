#!/usr/bin/env python
# adapted from https://github.com/microsoft/landcover/blob/master/training/pytorch/models/unet.py
import torch
import torch.nn as nn
import json
import os
import numpy as np

class Down(nn.Module):
    """
    Down blocks in U-Net
    """
    def __init__(self, conv, max):
        super(Down, self).__init__()
        self.conv = conv
        self.max = max

    def forward(self, x):
        x = self.conv(x)
        return self.max(x), x, x.shape[2]


class Up(nn.Module):
    """
    Up blocks in U-Net

    Similar to the down blocks, but incorporates input from skip connections.
    """
    def __init__(self, up, conv):
        super(Up, self).__init__()
        self.conv = conv
        self.up = up

    def forward(self, x, conv_out, D):
        x = self.up(x)
        lower = int(0.5 * (D - x.shape[2]))
        upper = int(D - lower)
        conv_out_ = conv_out[:, :, lower:upper, lower:upper] # adjust to zero padding
        x = torch.cat([x, conv_out_], dim=1)
        return self.conv(x)


class Unet(nn.Module):

    def __init__(self, model_opts):
        self.opts = model_opts['duke_unet_opts']
        super(Unet, self).__init__()
        self.n_input_channels = self.opts["n_input_channels"]
        self.n_classes = self.opts["n_classes"]
        self.initial_filters = self.opts["NUMBER_OF_FILTERS"]

        # down transformations
        max2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_1 = Down(self.conv_block(self.n_input_channels, self.initial_filters), max2d)
        self.down_2 = Down(self.conv_block(self.initial_filters, self.initial_filters*2), max2d)
        self.down_3 = Down(self.conv_block(self.initial_filters*2, self.initial_filters*4), max2d)
        self.down_4 = Down(self.conv_block(self.initial_filters*4, self.initial_filters*8), max2d)

        # midpoint
        self.conv5_block = self.conv_block(self.initial_filters*8, self.initial_filters*16)

        # up transformations
        conv_tr = lambda x, y: nn.ConvTranspose2d(x, y, kernel_size=2, stride=2)
        self.up_1 = Up(conv_tr(self.initial_filters*16, self.initial_filters*8), self.conv_block(self.initial_filters*16, self.initial_filters*8))
        self.up_2 = Up(conv_tr(self.initial_filters*8, self.initial_filters*4), self.conv_block(self.initial_filters*8, self.initial_filters*4))
        self.up_3 = Up(conv_tr(self.initial_filters*4, self.initial_filters*2), self.conv_block(self.initial_filters*4, self.initial_filters*2))
        self.up_4 = Up(conv_tr(self.initial_filters*2, self.initial_filters), self.conv_block(self.initial_filters*2, self.initial_filters))

        # Final output
        self.conv_final = nn.Conv2d(in_channels=self.initial_filters, out_channels=self.n_classes,
                                    kernel_size=1, padding=0, stride=1)


    def conv_block(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=True):
        """
        This is the main conv block for Unet. Two conv2d
        :param dim_in:
        :param dim_out:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        """

        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            #nn.ReLU(inplace=True)
            nn.LeakyReLU(0.01, inplace=True)
        )


    def fun(self, x):
        # down layers
        x, self.conv1_out, self.conv1_dim = self.down_1(x)
        x, self.conv2_out, self.conv2_dim = self.down_2(x)
        x, self.conv3_out, self.conv3_dim = self.down_3(x)
        x, self.conv4_out, self.conv4_dim = self.down_4(x)

        # Bottleneck
        x = self.conv5_block(x)

        return x
    
    def forward_partial(self, x):
        # up layers
        x = self.up_1(x, self.conv4_out, self.conv4_dim)
        x = self.up_2(x, self.conv3_out, self.conv3_dim)
        x = self.up_3(x, self.conv2_out, self.conv2_dim)
        x = self.up_4(x, self.conv1_out, self.conv1_dim)
        return self.conv_final(x)



    def forward(self, x):
        # down layers
        x, conv1_out, conv1_dim = self.down_1(x)
        x, conv2_out, conv2_dim = self.down_2(x)
        x, conv3_out, conv3_dim = self.down_3(x)
        x, conv4_out, conv4_dim = self.down_4(x)

        # Bottleneck
        x = self.conv5_block(x)

        # up layers
        x = self.up_1(x, conv4_out, conv4_dim)
        x = self.up_2(x, conv3_out, conv3_dim)
        x = self.up_3(x, conv2_out, conv2_dim)
        x = self.up_4(x, conv1_out, conv1_dim)
        return self.conv_final(x)

