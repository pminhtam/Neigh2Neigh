from torch import nn
import numpy as np
from skimage import measure
import torch.nn.functional as F
import torch
from torch import nn


class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """

    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


# Exponential K function: sigmoid(x) * (1 - sigmoid(x))
#
class SoftHistogram(nn.Module):
    def __init__(self, bins=256, _min=0, _max=255, bandwidth=1,device="cuda", kernel='exponential'):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self._min = _min
        self._max = _max
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.delta = float(_max - _min) / float(bins)
        self.centers = float(_min) + self.delta * (torch.arange(bins).float().to(device) + 0.5)  # (256,)

    def forward2D(self, x):
        bs, length = x.size()
        x = x.unsqueeze(x, 1)  # (bs, 1, length)
        centers = self.centers.unsqueeze(0).repeat(bs, 1).unsqueeze(2)  # (bs, 256, 1)
        x = x - centers  # (bs, 256, length)

        # (bs, 256, length)
        if self.kernel == 'exponential':
            x = torch.sigmoid(self.bandwidth * (x + self.delta / 2)) - \
                torch.sigmoid(self.bandwidth * (x - self.delta / 2))
        elif self.kernel == 'gaussian':
            x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta

        x = x.sum(dim=-1)  # (bs, 256)
        x = x / x.sum(dim=-1)
        return x

    def forward3D(self, x):
        bs, c, length = x.size()  # length = im_size*im_size
        x = x.unsqueeze(2)  # (bs, c, 1, length)
        # print(x.size())
        # print(self.centers)
        centers = self.centers.repeat(bs, c, 1).unsqueeze(-1)  # (bs, c, 256, length)
        x = x - centers  # (bs, c, 256, length)
        # print(x.size())

        # ((bs, c, 256, length)
        if self.kernel == 'exponential':
            x = torch.sigmoid(self.bandwidth * (x + self.delta / 2)) - \
                torch.sigmoid(self.bandwidth * (x - self.delta / 2))
        elif self.kernel == 'gaussian':
            x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta

        x = x.sum(dim=-1)  # (bs, c, 256)
        # print(x)
        x = x / x.sum(dim=-1, keepdim=True)
        # print(x)
        # print(x.size())
        return x

    def forward(self, x):
        len_shape = len(x.size())
        # print(x.size())     # (bs, 3, 256*256)
        # print(len_shape)
        if len_shape == 2:
            return self.forward2D(x)
        elif len_shape == 3:
            return self.forward3D(x)
        else:
            return self.forward3D(x)


class EarthMoveDistance(nn.Module):
    """EarthMoveDistance"""

    def __init__(self, bandwidth=1,device="cuda"):
        super(EarthMoveDistance, self).__init__()
        self.soft_hist = SoftHistogram(bandwidth=bandwidth,device=device).to(device)

    def forward(self, x, y):
        # print(x)
        bs, c, _, _ = x.shape
        # (bs, 3, H, W)
        error_est = x
        error_est = error_est.clamp(0, 1) * 255
        error_est = error_est.view(bs, c, - 1)  # (bs, 3, im_size*im_size)
        error_est_hist = self.soft_hist(error_est)  # (bs, 3, 256)
        error_est_hist_cdf = torch.cumsum(error_est_hist, dim=-1)

        # (bs, 3, H, W)
        error_gt = y
        error_gt = error_gt.clamp(0, 1) * 255
        error_gt = error_gt.view(bs, c, -1)  # (bs, 3, 256)
        error_gt_hist = self.soft_hist(error_gt)
        error_gt_hist_cdf = torch.cumsum(error_gt_hist, dim=-1)

        dist = (error_est_hist_cdf - error_gt_hist_cdf) ** 2  # (bs, 3, 256)
        return torch.mean(torch.sum(dist, dim=[2]))