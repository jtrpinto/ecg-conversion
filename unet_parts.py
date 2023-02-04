'''
Pytorch U-Net parts code from https://github.com/milesial/Pytorch-UNet

'''


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 5, stride=1, padding=2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels, 5, stride=1, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvTranspose(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, mid_channels, 5, stride=1, padding=2, output_padding=0),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.ConvTranspose1d(mid_channels, out_channels, 5, stride=1, padding=2, output_padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(5),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=5, mode='linear', align_corners=True)
            self.conv = DoubleConvTranspose(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.Upsample(scale_factor=5, mode='nearest') 
            self.conv = DoubleConvTranspose(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                   diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, 5, stride=1, padding=2, output_padding=0),
            nn.Tanh(),
        )
        
    def forward(self, x):
        return self.conv(x)
