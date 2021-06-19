import torch
import argparse
import os
import torch
import torch.optim
import numpy as np
import argparse

from torch.utils import data

from ssdn.network import NoiseNetwork
from ssdn.Discriminator import DiscriminatorLinear
from lossfunction_dual import *
from datasets.DenoisingDatasets import BenchmarkTrain, SIDD_VAL
import torch.nn.functional as F
import torch.nn as nn
import time
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

import cv2
from skimage.metrics import peak_signal_noise_ratio
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import matplotlib.pyplot as plt

def test(config):
    # Train in CPU
    if config.gpu_id == -1:
        device = torch.device('cpu')
    # Train in GPU
    else:
        device = torch.device('cuda')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    batch_size = config.batch_size
    num_workers = config.num_workers

    # val_dataset = SIDD_VAL('../validation/RAW/')
    val_dataset = SIDD_VAL('/vinai/tampm2/SIDD')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)


    net = NoiseNetwork(out_channels=8).to(device)
    model_dis = DiscriminatorLinear(in_chn=4).to(device)

    if config.pretrain_model:
        print('Loading pretrained model.')
        checkpoint = torch.load(config.pretrain_model)
        net.load_state_dict(checkpoint['model_state_dict'])
        model_dis.load_state_dict(checkpoint['model_dis_state_dict'])
        print('Loaded pretrained model sucessfully.')

    with torch.no_grad():
        psnr_ori = []
        psnr = []
        psnr2 = []
        psnr_mean = []
        for ii, data in enumerate(val_loader):
            im_noisy = data[0].to(device)
            im_gt = data[1].to(device)
            out = net(im_noisy)
            restored = torch.clamp(out[:, :4, :, :], 0, 1)
            noise = torch.clamp(out[:, 4:, :, :], -1, 1)
            restored_2 = torch.clamp(im_noisy - noise, 0, 1)
            restored = np.transpose(restored.cpu().numpy(), [0, 2, 3, 1])
            restored_2 = np.transpose(restored_2.cpu().numpy(), [0, 2, 3, 1])
            im_gt_np = np.transpose(im_gt.cpu().numpy(), [0, 2, 3, 1])
            im_noisy_np = np.transpose(im_noisy.cpu().numpy(), [0, 2, 3, 1])

            restored_mean = (restored + restored_2)/2
            psnr_ori.extend(
                [peak_signal_noise_ratio(im_noisy_np[i], im_gt_np[i], data_range=1) for i in range(batch_size)])
            psnr.extend(
                [peak_signal_noise_ratio(restored[i], im_gt_np[i], data_range=1) for i in range(batch_size)])
            psnr2.extend([peak_signal_noise_ratio(restored_2[i], im_gt_np[i], data_range=1) for i in range(batch_size)])
            psnr_mean.extend([peak_signal_noise_ratio(restored_mean[i], im_gt_np[i], data_range=1) for i in range(batch_size)])

        print('psnr_ori={:.4e}  ,psnr={:.4e}  ,  psnr2={:.4e} ,psnr_mean={:.4e} '.format(np.mean(psnr_ori),np.mean(psnr), np.mean(psnr2),np.mean(psnr_mean)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--pretrain_model', type=str, default='model_dual_34_gan/model_32.pth')
    # parser.add_argument('--pretrain_model', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=16)

    config = parser.parse_args()
    test(config)
