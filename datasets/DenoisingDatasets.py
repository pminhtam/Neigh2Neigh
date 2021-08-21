#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import torch
import glob
import random
import scipy.io as sio
import os
import numpy as np
# import cv2
from .data_tools import random_augmentation
from . import BaseDataSetH5


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

        input_blue = mask_blue * im_noisy
        input_blue = input_blue[0::2,0::2] + input_blue[0::2,1::2] + \
                     input_blue[1::2,0::2] + input_blue[1::2,1::2]
        # input_red, input_blue = random_augmentation(input_red, input_blue)


        # data augmentation
        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        input_red = torch.from_numpy(input_red.transpose((2, 0, 1)))
        input_blue = torch.from_numpy(input_blue.transpose((2, 0, 1)))
  
        mask_red = torch.from_numpy(mask_red.transpose((2, 0, 1)))
        mask_blue = torch.from_numpy(mask_blue.transpose((2, 0, 1)))

        return im_noisy, im_gt, input_red, input_blue, mask_red, mask_blue,im_folder


    def randomize(self, H, W):
        mask = np.zeros((H, W))
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
    
    def create_mask(self, y1, y2):  # dùng để aware cái weight
        bilateralFilter = lambda tensor: np.stack([cv2.bilateralFilter(tensor[...,i], 9, 75, 75) for i in range(4)], axis=-1)
        Laplacian = lambda tensor: np.stack([cv2.Laplacian(tensor[...,i], -1, ksize=5) for i in range(4)], axis=-1)

        int_y1 = (y1*255).astype(np.uint8)
        int_y2 = (y2*255).astype(np.uint8)

        int_y1 = bilateralFilter(y1)
        int_y2 = bilateralFilter(y2)

        int_y1 = Laplacian(int_y1).astype(np.int32)
        int_y2 = Laplacian(int_y2).astype(np.int32)

        mask = np.abs(int_y1 - int_y2)
        mask = np.exp(-mask**2 / (2*75.0))

        return mask




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
map_table = {
    'GP': 'bggr',
    'IP': 'rggb',
    'S6': 'grbg',
    'N6': 'bggr',
    'G4': 'bggr'
}

sence_name = ['0009_001_S6_00800_00350_3200_L',
                '0021_001_GP_10000_05000_5500_N',
                '0024_001_N6_03200_01500_5500_N',
                '0026_001_G4_00400_00160_5500_L',
                '0031_001_IP_01600_02000_3200_N',
                '0037_002_GP_06400_03200_3200_L',
                '0041_002_IP_01600_04000_5500_L',
                '0046_002_G4_00400_00350_3200_L',
                '0049_002_N6_00800_00800_5500_L',
                '0053_002_S6_03200_02000_5500_N',
                '0056_003_N6_03200_04000_5500_N',
                '0058_003_G4_00400_00500_5500_L',
                '0061_003_S6_00800_00500_4400_L',
                '0067_003_GP_01600_01500_3200_L',
                '0071_003_IP_02000_04000_5500_L',
                '0074_004_N6_00400_00040_3200_L',
                '0079_004_G4_00800_00160_3200_N',
                '0082_004_S6_03200_00500_4400_L',
                '0085_004_GP_06400_02000_4400_N',
                '0093_004_IP_01250_00250_3200_L',
                '0095_005_N6_00400_00200_3200_L',
                '0100_005_G4_00800_00400_3200_N',
                '0103_005_S6_01600_00800_4400_L',
                '0109_005_GP_10000_08000_4400_N',
                '0112_005_IP_01000_07500_5500_L',
                '0119_006_N6_00400_00100_3200_L',
                '0124_006_G4_00800_00350_3200_N',
                '0128_006_S6_03200_01600_4400_L',
                '0131_006_GP_01600_01250_4400_N',
                '0141_006_IP_01600_01620_3200_L',
                '0143_007_N6_00800_00800_4400_N',
                '0148_007_G4_00400_00400_4400_L',
                '0153_007_S6_03200_03200_5500_L',
                '0158_007_GP_03200_03200_5500_N',
                '0162_007_IP_01600_01600_3200_L',
                '0171_008_N6_03200_01600_4400_L',
                '0174_008_G4_00800_00800_4400_N',
                '0176_008_S6_00400_00100_5500_L',
                '0183_008_GP_06400_06400_5500_N',
                '0187_008_IP_01600_01600_3200_L'
]



def tensor2raw(tensor, bayer_pattern):
    r = tensor[...,0]
    g1 = tensor[...,1]
    g2 = tensor[...,2]
    b = tensor[...,3]
    h, w = r.shape[:2]
    raw = np.zeros((h*2,w*2))
    
    if bayer_pattern.lower() == 'bggr':
        raw[1::2,1::2] = r
        raw[0::2,1::2] = g1
        raw[1::2,0::2] = g2
        raw[0::2,0::2] = b
    elif bayer_pattern.lower() == 'rggb':
        raw[0::2,0::2] = r
        raw[0::2,1::2] = g1
        raw[1::2,0::2] = g2
        raw[1::2,1::2] = b
    elif bayer_pattern.lower() == 'grbg':
        raw[0::2,1::2]= r
        raw[0::2,0::2] = g1
        raw[1::2,1::2] = g2
        raw[1::2,0::2] = b
    else:
        raise Exception("Sorry, this bayer pattern isnot processed")
    
    return raw

class SIDD_VAL(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.noisy, self.clean = self.load_image(data_dir)
        self.num_image, self.num_boxes = self.noisy.shape[:2]

        self.len_dataset =  self.num_image * self.num_boxes
        print(self.len_dataset)


    def __getitem__(self, index):
        image_idx = index // self.num_boxes
        box_idx = index % self.num_boxes

        # Get device
        device = sence_name[image_idx][9:11]
        pattern = map_table[device]
        
        # Read images
        noisy = self.noisy[image_idx][box_idx]
        noisy = self.raw2tensor(noisy, pattern)
        noisy = self.numpy2torch(noisy)

        clean = self.clean[image_idx][box_idx]
        clean = self.raw2tensor(clean, pattern)
        clean = self.numpy2torch(clean)

        return noisy, clean


    def __len__(self):
        return self.len_dataset


    def load_image(self, data_dir):
        noisy = sio.loadmat(os.path.join(data_dir, 'ValidationNoisyBlocksRaw.mat'))
        gt = sio.loadmat(os.path.join(data_dir, 'ValidationGtBlocksRaw.mat'))

        return noisy['ValidationNoisyBlocksRaw'], gt['ValidationGtBlocksRaw']
      

    def numpy2torch(self, img):
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        return img_tensor

    def raw2tensor(self, raw, bayer_pattern):
        if bayer_pattern.lower() == 'bggr':
            r = raw[1::2,1::2]
            g2 = raw[0::2,1::2]
            g1 = raw[1::2,0::2]
            b = raw[0::2,0::2]
        elif bayer_pattern.lower() == 'rggb':
            r = raw[0::2,0::2]
            g1 = raw[0::2,1::2]
            g2 = raw[1::2,0::2]
            b = raw[1::2,1::2]
        elif bayer_pattern.lower() == 'grbg':
            r = raw[0::2,1::2]
            g1 = raw[0::2,0::2]
            g2 = raw[1::2,1::2]
            b = raw[1::2,0::2]
        else:
            raise Exception("Sorry, this bayer pattern isnot processed")
        
        return np.stack([r,g1,g2,b], axis=-1)