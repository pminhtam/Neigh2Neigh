#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:35:24

import random
# import cv2
# import numpy as np
import torch.utils.data as uData
# import h5py
import os

class BaseDataSetH5(uData.Dataset):
    def __init__(self, folder_path, length=None):

        super(BaseDataSetH5, self).__init__()
        self.folder_path = folder_path
        self.length = length

        self.num_images = len(os.listdir(folder_path))

    def __len__(self):
        if self.length == None:
            return self.num_images
        else:
            return self.length

    def crop_patch(self, imgs_sets):
        H, W = imgs_sets[0].shape[:2]
        # print(H, W)
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
       
        return [im[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size] for im in imgs_sets]
