#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import torch
import torch.nn as nn

class Simplify_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Simplify_Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(6)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):

        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x
