# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG

class UNET_3D(nn.Module):

    def __init__(self):

        super().__init__()
        #encoder
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3_1 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4_1 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        #middle
        self.conv5_1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(1024)

        #decoder
        self.deconv6_1 = nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv6_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv6_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv6_4 = nn.Conv3d(512, 256, kernel_size=3, padding=1)

        self.deconv7_1 = nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv7_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv7_3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv7_4 = nn.Conv3d(256, 128, kernel_size=3, padding=1)

        self.deconv8_1 = nn.ConvTranspose3d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv8_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv8_3 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv8_4 = nn.Conv3d(128, 64, kernel_size=3, padding=1)

        self.deconv9_1 = nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv9_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv9_3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv9_4 = nn.Conv3d(64, 1, kernel_size=1)
        self.outputer = nn.Tanh()

    def forward(self, x):

        #encoder
        #in 256x256
        y = self.bn1(self.relu(self.conv1_1(x)))
        y1 = self.bn1(self.relu(self.conv1_2(y)))
        y = self.pool(y1)
        #in 128x128
        y = self.bn1(self.relu(self.conv2_1(y)))
        y = self.bn2(self.relu(self.conv2_2(y)))
        y2 = self.bn2(self.relu(self.conv2_3(y)))
        y = self.pool(y2)
        #in 64x64
        y = self.bn2(self.relu(self.conv3_1(y)))
        y = self.bn3(self.relu(self.conv3_2(y)))
        y3 = self.bn3(self.relu(self.conv3_3(y)))
        y = self.pool(y3)
        #in 32x32
        y = self.bn3(self.relu(self.conv4_1(y)))
        y = self.bn4(self.relu(self.conv4_2(y)))
        y4 = self.bn4(self.relu(self.conv4_3(y)))
        y = self.pool(y4)
        #in 16x16
        #middle
        y = self.bn4(self.relu(self.conv5_1(y)))
        y = self.bn5(self.relu(self.conv5_2(y)))
        y = self.bn4(self.relu(self.conv5_3(y)))

        #decoder


        y = self.bn4(self.relu(self.deconv6_1(y)))
        y = y + y4
        y = self.bn4(self.relu(self.conv6_2(y)))
        y = self.bn4(self.relu(self.conv6_3(y)))
        y = self.bn3(self.relu(self.conv6_4(y)))


        y = self.bn3(self.relu(self.deconv7_1(y)))
        y = y + y3
        y = self.bn3(self.relu(self.conv7_2(y)))
        y = self.bn3(self.relu(self.conv7_3(y)))
        y = self.bn2(self.relu(self.conv7_4(y)))


        y = self.bn2(self.relu(self.deconv8_1(y)))
        y = y + y2
        y = self.bn2(self.relu(self.conv8_2(y)))
        y = self.bn2(self.relu(self.conv8_3(y)))
        y = self.bn1(self.relu(self.conv8_4(y)))


        y = self.bn1(self.relu(self.deconv9_1(y)))
        y = y + y1
        y = self.bn1(self.relu(self.conv9_2(y)))
        y = self.bn1(self.relu(self.conv9_3(y)))
        y = self.outputer(self.relu(self.conv9_4(y)))

        return y



