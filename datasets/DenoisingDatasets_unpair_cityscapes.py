#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import torch
import glob
import random
import scipy.io as sio
import os
import numpy as np
import cv2
from .data_tools import random_augmentation
from . import BaseDataSetH5
from PIL import Image

## Use unpair clean to denoise

# Benchmardk Datasets: Renoir and SIDD
class BenchmarkTrain(BaseDataSetH5):
    def __init__(self, noise_dir, gt_dir, length, pch_size=1024):
        super(BenchmarkTrain, self).__init__(noise_dir, length)
        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.list_pairs = glob.glob(os.path.join(noise_dir, '*.png'))
        self.num_images = len(self.list_pairs)
        print(len(self.list_pairs))
        self.noise_path = []
        self.noise_path.extend(self.list_pairs)

        self.pch_size = pch_size
        patterns = [ np.array( [ [1, 2], [0, 0] ] ),
                     np.array( [ [2, 1], [0, 0] ] ),

                     np.array( [ [1, 0], [2, 0] ] ),
                     np.array( [ [2, 0], [1, 0] ] ),

                     np.array( [ [1, 0], [0, 2] ] ),
                     np.array( [ [2, 0], [0, 1] ] ),
                     
                     np.array( [ [0, 0], [1, 2] ] ),
                     np.array( [ [0, 0], [2, 1] ] ),

                     np.array( [ [0, 1], [2, 0] ] ),
                     np.array( [ [0, 2], [1, 0] ] ),
                     
                     np.array( [ [0, 1], [0, 2 ]] ),
                     np.array( [ [0, 2], [0, 1] ] ),]

        self.patterns = np.stack(patterns, axis=0)


    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)
        # ind_im = index
        # ind_im2 = random.randint(0, num_images-1)
        ind_im2 = (ind_im + 1)% (num_images-1)
        # while ind_im2 == ind_im:
        #     ind_im2 = random.randint(0, num_images - 1)
        # im_folder = self.list_pairs[ind_im]
        # im_folder2 = self.list_pairs[ind_im2]
        im_folder = self.noise_path[ind_im]
        # noise_data = sio.loadmat(os.path.join(im_folder, 'noisy.mat'))
        # gt_data = sio.loadmat(os.path.join(im_folder, 'clean.mat'))
        # gt_2 = sio.loadmat(os.path.join(im_folder2, 'clean.mat'))

        # im_noisy = noise_data['x']
        # im_gt = gt_data['x']
        # im_gt2 = gt_2['x']
        # print(self.noise_path[0])
        # try:
        im_noisy = np.array(Image.open(self.noise_path[ind_im]).convert('RGB'),np.float32)/255.0
        im_gt = np.array(Image.open(os.path.join(self.gt_dir, self.noise_path[ind_im].split("/")[-1])).convert('RGB'),np.float32)/255.0
        im_gt2 = np.array(Image.open(os.path.join(self.gt_dir, self.noise_path[ind_im2].split("/")[-1])).convert('RGB'),np.float32)/255.0
        # except:
        #     print(ind_im)
        #     print(ind_im2)
        #     print(self.noise_path[ind_im])
            # print(os.path.join(self.gt_dir, self.noise_path[ind_im].split("/")[-1]))
            # print(os.path.join(self.gt_dir, self.noise_path[ind_im2].split("/")[-1]))

        im_gt, im_noisy = self.crop_patch([im_gt, im_noisy])
        im_gt2 = self.crop_patch([im_gt2])[0]
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)
        # print(im_gt2)
        im_gt2 = random_augmentation(im_gt2)[0]
        # print(im_gt2.shape)

        mask_red, mask_blue = self.randomize(self.pch_size, self.pch_size,3)

        input_red = mask_red * im_noisy
        input_red = input_red[0::2,0::2] + input_red[0::2,1::2] + \
                    input_red[1::2,0::2] + input_red[1::2,1::2]
        input_blue = mask_blue * im_noisy
        input_blue = input_blue[0::2,0::2] + input_blue[0::2,1::2] + \
                     input_blue[1::2,0::2] + input_blue[1::2,1::2]
        # data augmentation
        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        input_red = torch.from_numpy(input_red.transpose((2, 0, 1)))
        input_blue = torch.from_numpy(input_blue.transpose((2, 0, 1)))
  
        mask_red = torch.from_numpy(mask_red.transpose((2, 0, 1)))
        mask_blue = torch.from_numpy(mask_blue.transpose((2, 0, 1)))
        im_gt2 = torch.from_numpy(im_gt2.transpose((2, 0, 1)))
        # print(im_gt.shape)
        return im_noisy, im_gt, input_red, input_blue, mask_red, mask_blue,im_folder,im_gt2


    def randomize(self, H, W,C):
        pattern_mask = np.random.randint(0, 12, (H//2, W//2))
        mask = np.zeros((H, W))
        for h in range(H // 2):
            for w in range(W // 2):
                h1 = h*2 ; h2 = h1 + 2
                w1 = w*2 ; w2 = w1 + 2
                mask[h1:h2,w1:w2] = self.patterns[pattern_mask[h, w]]

        mask1 = (mask==1).astype(np.float32) ; mask1 = np.stack([mask1]*C, axis=-1)
        # mask2 = 1 - mask1
        mask2 = (mask==2).astype(np.float32) ; mask2 = np.stack([mask2]*C, axis=-1)

        return mask1, mask2
    



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


