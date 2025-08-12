import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from PIL import Image

import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x
    
class ImprovedUNet(nn.Module):
    def __init__(self):
        super(ImprovedUNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.middle = conv_block(256, 512)
        self.up3 = up_block(512, 256)
        self.up2 = up_block(256, 128)
        self.up1 = up_block(128, 64)

        self.final = nn.Conv2d(64, 3, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.middle(self.pool(e3))

        d3 = self.up3(m) + e3
        d2 = self.up2(d3) + e2
        d1 = self.up1(d2) + e1

        out = self.final(d1)
        return self.output_activation(out)
    
from torchvision.models import vgg16
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.l1 = nn.L1Loss()

    def forward(self, input, target):
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)
        return self.l1(input, target) + 0.1 * self.l1(input_vgg, target_vgg)



class BokehDataset(Dataset):
    def __init__(self, orig_folder, bokeh_folder, transform=None):
        self.orig_folder = orig_folder
        self.bokeh_folder = bokeh_folder
        self.transform = transform
        self.images = [f for f in os.listdir(orig_folder) if os.path.isfile(os.path.join(orig_folder, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        orig_path = os.path.join(self.orig_folder, self.images[idx])
        bokeh_path = os.path.join(self.bokeh_folder, self.images[idx])
        orig_img = Image.open(orig_path).convert('RGB')
        bokeh_img = Image.open(bokeh_path).convert('RGB')
        if self.transform:
            orig_img = self.transform(orig_img)
            bokeh_img = self.transform(bokeh_img)
        return orig_img, bokeh_img