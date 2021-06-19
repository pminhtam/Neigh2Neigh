import torch
from torch import nn
import torch.nn.functional as F
from subnet import UNet


class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        self.unet = UNet(in_channels=4, out_channels=4, wf=64)
    def forward(self, noisy):   
        return self.unet(noisy) + noisy
