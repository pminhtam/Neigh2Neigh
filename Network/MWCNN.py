import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, depth=4, wf=64, slope=0.2):
        """
        Reference:
        Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical
        Image Segmentation. MICCAI 2015.
        ArXiv Version: https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 4
            wf (int): number of filters in the first layer, Default 32
        """
        super(UNet, self).__init__()
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(UNetConvBlock(prev_channels, (2**i)*wf, slope))
            self.down_path.append(
                nn.Sequential(
                    DWTForward(),
                    nn.Conv2d(4*(2**i)*wf, (2**i)*wf, kernel_size=3, padding=1, bias=True)
                )
            )
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i)*wf

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        blocks = []
        for i, (encode, down) in enumerate(zip(self.encoder, self.down_path)):
            x = encode(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = down(x)
                

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=True))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetUpBlock, self).__init__()
        # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.up = nn.Sequential(
            DWTInverse(),
            nn.Conv2d(in_size//4, out_size, 3, 1, 1)
        )
        self.conv_block = UNetConvBlock(in_size, out_size, slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class DWTForward(nn.Module):
    def __init__(self):
        super(DWTForward, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class DWTInverse(nn.Module):
    def __init__(self):
        super(DWTInverse, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                            hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y

class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        self.unet = UNet(in_channels=4, out_channels=4, wf=64)
    def forward(self, noisy):   
        return self.unet(noisy) + noisy
