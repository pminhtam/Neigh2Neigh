from torch import nn
import torch.nn.functional as F
import torch

class BasicLoss(nn.Module):
    def __init__(self):
        super(BasicLoss, self).__init__()


    def forward(self,im_noisy,im_restore,im_restore_noise, input_red, denoise_red, noise_red, input_blue, denoise_blue, noise_blue):

        # mse = torch.abs(denoise_red - denoise_blue)**2 # 1
        # # mse =  mse_img**2
        # mse_all =  (im_noisy - im_restore)**2  # 2

        # r1 = torch.abs(out_red - in_blue)
        mse_red =  (input_red - denoise_blue)**2 #3
        mse_blue =  (input_blue - denoise_red)**2

        # r_red =  (input_red - (noise_red + denoise_red))**2  # 4
        # r_blue =  (input_blue - noise_blue - denoise_blue)**2
        # # print(im_restore_noise.size())
        # mse_all_noise =  (im_noisy - im_restore - im_restore_noise)**2  # 5

        alpha = 0.5
        beta = 0.1
        # return torch.mean(0.2*mse + 0.7*(mse_red + mse_blue) + alpha*(r_red + r_blue)) + 0.5*torch.mean(mse_all)  # 1234_change_role_4  # 50.1
        # return torch.mean(0.7*(mse_red + mse_blue) + alpha*(r_red + r_blue)) + 0.5*torch.mean(mse_all)  # 234_change_role_4  # 50.1
        # return torch.mean((r_red + r_blue)) + torch.mean(mse_all)  # 24_change_role_4  # 27
        # return torch.mean(0.2*mse + 0.7*(mse_red + mse_blue)) + 0.5*torch.mean(mse_all)  # 123_change_role_3 # 50.28
        # return torch.mean(0.2*mse + 0.7*(r_red + r_blue)) + 0.5*torch.mean(mse_all) + 0.5*torch.mean(mse_all_noise)  # 1245
        # return 0.5*torch.mean(mse_all)  # 2  # 37.6

        # return torch.mean(mse_red + mse_blue)  # 3  # 50.29

        # return torch.mean(r_red + r_blue)  # 4 # 27.8
        # return torch.mean(mse)  # 1  # 20
        return torch.mean(mse_red + mse_blue)  # 3_not_change_role  # 37.3
        # return torch.mean(mse_red + mse_blue+ r_red + r_blue)  # 34_gan # 49.9
        # return torch.mean(mse_red + mse_blue+ r_red + r_blue) + torch.mean(mse_all_noise)  # 345_gan   # 49.9
        # return 0.5*torch.mean(mse + alpha*(r_red + r_blue)) + torch.mean(mse_all)
        # return torch.mean(mse_all_noise) + torch.mean(mse_all) # 15



class BasicLoss_fake(nn.Module):
    def __init__(self):
        super(BasicLoss_fake, self).__init__()


    def forward(self,denoise_red,noise_blue,denoise_blue,noise_red,denoise_fake_red,noise_fake_red,denoise_fake_blue,noise_fake_blue):

        red = torch.abs(denoise_red - denoise_fake_red)**2 #
        noise_red = torch.abs(noise_blue - noise_fake_red)**2 #
        blue = torch.abs(denoise_blue - denoise_fake_blue)**2 # 1
        noise_blue = torch.abs(noise_red - noise_fake_blue)**2 #

        return torch.mean(red+noise_red+blue+noise_blue)

class FourierLoss(nn.Module):
    def __init__(self):
        super(FourierLoss, self).__init__()


    def forward(self,noise_blue,noise_red):
        noise_red = rfft(noise_red)
        noise_blue = rfft(noise_blue)
        rmse = torch.abs(noise_blue - noise_red)**2 #

        return torch.mean(rmse)

def rfft(x: torch.tensor):
    """Applies torch.rfft to input (in 3 dimensions).
    Also permutes fourier space in front of c x w x h.
    i.e. input shape: b x c x w x h -> output shape: b x 2 x c x w x h
    Args:
        x: tensor that should fourier transform applied to
    Returns:
        Fourier transform of input
    """
    # insert last dimension infront of c x w x h
    # b x c x w x h x fourier -> b x fourier x c x w x h
    original_permutation = range(len(x.shape))
    permute = [
        *original_permutation[:-3],
        len(original_permutation),
        *original_permutation[-3:],
    ]
    return torch.rfft(x, 3, onesided=False, normalized=False).permute(permute)
