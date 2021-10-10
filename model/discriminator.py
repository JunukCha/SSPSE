import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseDiscriminator(nn.Module):
    def __init__(self):
        super(BaseDiscriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)
        return x.view(-1, 1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*8, 1, 7, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)
        return x.view(-1, 1)


class JointsDiscriminator(nn.Module):
    def __init__(self):
        super(JointsDiscriminator, self).__init__()

        self.nc = 512
        self.main = nn.Sequential(
            nn.Linear(14*3, self.nc, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.nc, self.nc*2, bias=False),
            nn.BatchNorm1d(self.nc*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.nc*2, self.nc*4, bias=False),
            nn.BatchNorm1d(self.nc*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.nc*4, self.nc*8, bias=False),
            nn.BatchNorm1d(self.nc*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.nc*8, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 14*3)
        x = self.main(x)
        return x

class MotionDiscriminator(nn.Module):
    def __init__(self):
        super(MotionDiscriminator, self).__init__()
        self.fc1 = nn.Linear(69, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = x.view(-1, 69)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def get_discriminator(model):
    net = globals()[model]()
    return net
