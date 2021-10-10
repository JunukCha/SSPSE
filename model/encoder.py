import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import numpy as np

def get_resnet_encoder(pretrained=False):
    net = models.resnet50(pretrained=pretrained)
    return net


def get_alexnet_encoder():
    net = models.alexnet(pretrained=True)
    return net


class ContextEncoderNet(nn.Module):
    def __init__(self):
        super(ContextEncoderNet, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 64*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class JigsawPuzzleNet(nn.Module):
    def __init__(self):
        super(JigsawPuzzleNet, self).__init__()
        
        self.classes = 24
        self.num_patch = 5

        self.fc = nn.Sequential(
            nn.Linear(5*2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, self.classes),
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.classifier(x)
        return x
    
class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.classes = 4

        self.classifier = nn.Sequential(
            nn.Linear(2048, self.classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


def get_encoder(model, pretrained=False):
    if model == "resnet":
        net = get_resnet_encoder(pretrained)
    elif model == "alexnet":
        net = get_alexnet_encoder()
    else:
        net = globals()[model]()
    return net