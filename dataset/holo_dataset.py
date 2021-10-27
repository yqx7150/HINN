from __future__ import print_function, division
import os, random, time
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import rawpy
from glob import glob
from PIL import Image as PILImage
import numbers
from scipy.misc import imread
import cv2
import matplotlib.pyplot as plt

####################################train_dataset##########################################

class imageDataset(Dataset):
    def __init__(self, opt):
       
        root_phases='./train_192/phs'
        root_img='./train_192/amp'
        self.datanames_phases = np.array([root_phases+"/"+x  for x in os.listdir(root_phases)])
        self.datanames_img = np.array([root_img+"/"+x  for x in os.listdir(root_img)])
        
    def random_flip(self, image_rgb, image_phase):
        idx = np.random.randint(2)
        image_rgb = np.flip(image_rgb,axis=idx).copy()
        image_phase = np.flip(image_phase,axis=idx).copy()
        
        return image_rgb, image_phase

    def random_rotate(self, image_rgb, image_phase):
        idx = np.random.randint(4)
        image_rgb = np.rot90(image_rgb,k=idx)
        image_phase = np.rot90(image_phase,k=idx)

        return image_rgb, image_phase

    def random_crop(self, patch_size, image_rgb, image_phase):
        x=np.zeros([192,192,3])
        H, W, _ = x.shape
        rnd_h = random.randint(0, max(0, H - patch_size))
        rnd_w = random.randint(0, max(0, W - patch_size))
        patch_image_rgb = image_rgb[rnd_h:rnd_h +patch_size, rnd_w:rnd_w + patch_size, :]
        patch_image_phase = image_phase[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        
        return patch_image_rgb, patch_image_phase
        
    def aug(self, patch_size, image_rgb, image_phase):
        image_rgb_crop, image_phase_crop = self.random_crop(patch_size, image_rgb, image_phase)
        image_rgb_rotate, image_phase_rotate = self.random_rotate(image_rgb_crop, image_phase_crop)
        image_rgb_end, image_phase_end = self.random_flip(image_rgb_rotate, image_phase_rotate)
        
        return image_rgb_end, image_phase_end

    def norm_img(self, img, max_value):
        img = img / float(max_value)        
        return img

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)
        
    def __len__(self):
        return len(self.datanames_img)
    
    def __getitem__(self, index):
        image_rgb = cv2.imread(self.datanames_img[index],flags=-1)
        image_phase=cv2.imread(self.datanames_phases[index],flags=-1)
        self.patch_size=192
        image_rgb,image_phase = self.aug(self.patch_size, image_rgb, image_phase)
        image_rgb=(image_rgb-image_rgb.min())/(image_rgb.max()-image_rgb.min())
        image_phase=(image_phase-image_phase.min())/(image_phase.max()-image_phase.min())
        image_rgb = self.np2tensor(image_rgb)
        image_phase = self.np2tensor(image_phase)
        sample = {'input_raw':image_rgb, 'target_phase':image_phase, 'target_raw':image_rgb}
        return sample

#####################################test_dataset#########################################

class imageDataset_test(Dataset):
    def __init__(self, opt):
        root_phases='./test_data/phs'
        root_img='./test_data/amp'
        self.datanames_phases= np.array([root_phases+"/"+x  for x in os.listdir(root_phases)])
        self.datanames_img = np.array([root_img+"/"+x  for x in os.listdir(root_img)])
    

    def norm_img(self, img, max_value):
        img = img / float(max_value)        
        return img

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)
        
    def __len__(self):
        return len(self.datanames_img)
    
    def __getitem__(self, index):
        image_rgb = cv2.imread(self.datanames_img[index],flags=-1)
        image_rgb=abs(image_rgb)
        image_phase=cv2.imread(self.datanames_phases[index],flags=-1)
        image_rgb=(image_rgb-image_rgb.min())/(image_rgb.max()-image_rgb.min())
        image_phase=(image_phase-image_phase.min())/(image_phase.max()-image_phase.min())
        image_rgb = self.np2tensor(image_rgb)
        image_phase = self.np2tensor(image_phase)
        sample = {'input_raw':image_rgb, 'target_phase':image_phase, 'target_raw':image_rgb,'file_name':self.datanames_img[index].split("/")[-1].split(".")[0]}
        return sample


