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
## add mask of each small image in dataloader
## to process like Noise2Self

# Benchmardk Datasets: Renoir and SIDD
class BenchmarkTrain(BaseDataSetH5):
    def __init__(self, folder_path, length, pch_size=1024):
        super(BenchmarkTrain, self).__init__(folder_path, length)
        self.list_pairs = glob.glob(os.path.join(folder_path, '*'))
        print(len(self.list_pairs))
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
        im_folder = self.list_pairs[ind_im]
        
        noise_data = sio.loadmat(os.path.join(im_folder, 'noisy.mat'))
        gt_data = sio.loadmat(os.path.join(im_folder, 'clean.mat'))

        im_noisy = noise_data['x']
        im_gt = gt_data['x']

        im_gt, im_noisy = self.crop_patch([im_gt, im_noisy])
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        mask_red, mask_blue = self.randomize(self.pch_size, self.pch_size)

        input_red = mask_red * im_noisy
        input_red = input_red[0::2,0::2] + input_red[0::2,1::2] + \
                    input_red[1::2,0::2] + input_red[1::2,1::2]
        input_red_c = (1- mask_red) * im_noisy
        input_blue = mask_blue * im_noisy
        input_blue = input_blue[0::2,0::2] + input_blue[0::2,1::2] + \
                     input_blue[1::2,0::2] + input_blue[1::2,1::2]
        size = input_blue.shape
        # print(size)
        mask_j = np.random.randint(2, size=(size[0], size[1]))
        mask_j = np.expand_dims(mask_j,axis=2)
        mask_j = np.repeat(mask_j, size[2],2)
        # print(mask_j.shape)

        # input_red, input_blue = random_augmentation(input_red, input_blue)


        # data augmentation
        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        input_red = torch.from_numpy(input_red.transpose((2, 0, 1)))
        input_red_c = torch.from_numpy(input_red_c.transpose((2, 0, 1)))
        input_blue = torch.from_numpy(input_blue.transpose((2, 0, 1)))
  
        mask_red = torch.from_numpy(mask_red.transpose((2, 0, 1)))
        mask_blue = torch.from_numpy(mask_blue.transpose((2, 0, 1)))

        mask_j = torch.from_numpy(mask_j.transpose((2, 0, 1)))


        # input_red = input_red * mask_j
        # input_blue = input_blue * (1 - mask_j)

        return im_noisy, im_gt, input_red, input_blue, mask_red, mask_blue,im_folder,mask_j,input_red_c


    def randomize(self, H, W):
        pattern_mask = np.random.randint(0, 12, (H//2, W//2))
        mask = np.zeros((H, W))
        for h in range(H // 2):
            for w in range(W // 2):
                h1 = h*2 ; h2 = h1 + 2
                w1 = w*2 ; w2 = w1 + 2
                mask[h1:h2,w1:w2] = self.patterns[pattern_mask[h, w]]

        mask1 = (mask==1).astype(np.float32) ; mask1 = np.stack([mask1]*4, axis=-1)
        # mask2 = 1 - mask1
        mask2 = (mask==2).astype(np.float32) ; mask2 = np.stack([mask2]*4, axis=-1)

        return mask1, mask2
    



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


