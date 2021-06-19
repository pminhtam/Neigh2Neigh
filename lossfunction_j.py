from torch import nn
import torch.nn.functional as F
import torch
# loss with datasets/DenoisingDatasets_mask.py dataloader
## add mask like Noise2Void
class BasicLoss(nn.Module):
    def __init__(self):
        super(BasicLoss, self).__init__()


    def forward(self, input_red, denoise_red, input_blue, denoise_blue, mask_j):
        # print(input_red.shape)
        # print(denoise_blue.shape)
        # print(mask_j.shape)
        # r1 = torch.abs(out_red - in_blue)
        mse_red =  (input_red - denoise_blue * mask_j)**2 #3
        mse_blue =  (input_blue - denoise_red * (1-mask_j))**2

        return torch.mean(mse_red + mse_blue)



class BasicLoss_c(nn.Module):
    def __init__(self):
        super(BasicLoss_c, self).__init__()


    def forward(self,im_noisy, input_red, denoise_red_c,mask_red):
        # print(input_red.shape)
        # print(denoise_blue.shape)
        # print(mask_j.shape)
        # r1 = torch.abs(out_red - in_blue)
        mse_red_all =  (im_noisy - denoise_red_c)**2 #3
        denoise_red = F.max_pool2d(denoise_red_c * mask_red, kernel_size=2, stride=2)
        mse_red =  (input_red - denoise_red)**2 #3

        return torch.mean(mse_red_all) + torch.mean(mse_red)

