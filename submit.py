import sys
from typing import Pattern
sys.path.append('./')
import numpy as np
import torch
# from models import DenoiseNet
# from Network.RRG import DenoiseNet
# from ssdn.network import NoiseNetwork
from Network.MWCNN import DenoiseNet

from scipy.io import loadmat, savemat
import os
from tqdm import tqdm
import h5py
import torch

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

class SIDD_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.images = self.load_image(data_dir)
        self.num_image, self.num_boxes = self.images.shape[:2]

        self.len_dataset =  self.num_image * self.num_boxes
        print("Total training examples:", self.len_dataset)


    def __getitem__(self, index):
        image_idx = index // self.num_boxes
        box_idx = index % self.num_boxes

        # Get device
        device = sence_name[image_idx][9:11]
        pattern = map_table[device]
        
        # Read images
        noisy = self.images[image_idx][box_idx]
        noisy = self.raw2tensor(noisy, pattern)
        noisy = self.numpy2torch(noisy)

        return noisy, image_idx, box_idx, pattern


    def __len__(self):
        return self.len_dataset


    def load_image(self, data_dir):
        box_pos = loadmat(os.path.join(data_dir, 'BenchmarkNoisyBlocksRaw.mat'))
        return box_pos['BenchmarkNoisyBlocksRaw']
      

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

use_gpu = True

device = 'cuda' if use_gpu else 'cpu'


path = 'model_MWCNN/model_state_100.pth'
model_restoration = DenoiseNet().to(device)


print('Loading pretrained model.')
checkpoint = torch.load(path)
model_restoration.load_state_dict(checkpoint)
print('Loaded pretrained model sucessfully.')


data_dir = '../SIDD_Medium'
batch_size = 16
num_workers = 4

test_dataset = SIDD_Dataset(data_dir)
test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, 
                num_workers=num_workers, pin_memory=True, drop_last=False)

result = h5py.File('SubmitRaw.mat', 'r+')
with torch.autograd.set_grad_enabled(False):
    for data in tqdm(test_loader):
        noisy, image_idxes, box_idxes, pattern = data
        noisy = noisy.to(device)
        
        im_denoise = model_restoration(noisy)

        im_denoise = im_denoise.cpu().numpy()
        for i in range(len(image_idxes)):
            img_idx = image_idxes[i]
            box_idx = box_idxes[i]
            
            image = np.transpose(im_denoise[i], (1, 2, 0))
            image = np.clip(image, 0, 1)
            image = tensor2raw(image, pattern[i])

            f = result[result['DenoisedBlocksRaw'][box_idx][img_idx]] 
            f[...] = image.T

result.close()
