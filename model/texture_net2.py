import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class TextureNet(nn.Module):
    def __init__(self, tex_size):
        super(TextureNet, self).__init__()
        nz_feat = 200
        num_faces = 1280
        n_upconv = 5
        self.tex_size = tex_size

        self.nc_init = nc_init =256
        nc_final = 3
        img_H = int(2**np.floor(np.log2(np.sqrt(num_faces) * tex_size)))
        img_W = 2 * img_H
        self.feat_H = feat_H = img_H // (2 ** n_upconv)
        self.feat_W = feat_W = img_W // (2 ** n_upconv)
        
        self.resnet = models.resnet18(pretrained=True)
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc1 = nn.Linear(256*3*3, nz_feat)
        self.fc2 = nn.Linear(nz_feat, nz_feat)
        self.fc3 = nn.Linear(nz_feat, nc_init*feat_H*feat_W)
        self.fc4 = nn.Linear(nc_init*feat_H*feat_W, nc_init*feat_H*feat_W)

        mode = "bilinear"
        modules = []
        nc_output = nc_input = 256
        nc_step = 1
        nc_min = 8
        for nl in range(n_upconv):
            if (nl % nc_step==0) and (nc_output//2 >= nc_min):
                nc_output = nc_output//2
            modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(nc_input, nc_output, kernel_size=3, stride=1, padding=0),
                    nn.LeakyReLU(0.2,inplace=True)
                )
            )
            nc_input = nc_output
            modules.append(
                nn.Sequential(
                    nn.Conv2d(nc_input, nc_output, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(nc_output),
                    nn.LeakyReLU(0.2,inplace=True)
                )
            )

        modules.append(nn.Conv2d(nc_output, nc_final, kernel_size=3, stride=1, padding=1, bias=True))
        self.decoder = nn.Sequential(*modules)
        self.Tanh = nn.Tanh()

    def forward(self, x, uv_sampler):
        uv_sampler = uv_sampler.view(-1, 13776, self.tex_size*self.tex_size, 2)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.layer1(x)
        x = x.view(-1, 256*3*3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(-1, self.nc_init, self.feat_H, self.feat_W)
        x = self.decoder(x)
        x = self.Tanh(x)
        x = nn.functional.grid_sample(x, uv_sampler, align_corners=False)
        return x