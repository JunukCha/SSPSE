import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class TextureNet(nn.Module):
    def __init__(self, tex_size):
        super(TextureNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, 13776*3)
        # self.fc1 = nn.Linear(2048, 13776*1)
        # self.fc2 = nn.Linear(13776*1, 13776*3)
        self.tanh = nn.Tanh()
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)

        x = x.view(-1, 2048)
        x = self.fc(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.tanh(x)
        # x = self.ReLU(x)
        x = self.sigmoid(x)
        x = x.view(-1, 13776, 1, 1, 1, 3)
        return x
