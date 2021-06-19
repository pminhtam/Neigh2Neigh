from torch import nn
import torch.nn.functional as F
import torch
## Origin Neigh2Neigh
class BasicLoss(nn.Module):
    def __init__(self):
        super(BasicLoss, self).__init__()


    def forward(self, out_red, in_blue, im_restore, mask_red, mask_blue):
        red_restore  = F.max_pool2d(im_restore * mask_red, kernel_size=2, stride=2)
        blue_restore = F.max_pool2d(im_restore * mask_blue, kernel_size=2, stride=2)

        out_red = torch.clamp(out_red, 0, 1)
        red_restore = torch.clamp(red_restore, 0, 1)
        blue_restore = torch.clamp(blue_restore, 0, 1)

        mse_img = torch.abs(out_red - in_blue)
        mse =  mse_img**2
        # r1 = torch.abs(out_red - in_blue)
        r2 = torch.abs(red_restore - blue_restore)
        R =  (out_red - in_blue - red_restore + blue_restore)**2 

        return torch.mean(mse + R),mse,R