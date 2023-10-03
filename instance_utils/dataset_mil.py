from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import math
import pandas as pd
import os
import random
import cv2
import matplotlib.pyplot as plt
import sys
import gc
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from ocr_models.modules.text_localization import * 
class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img
    

class ILDataset(Dataset):
    def __init__(self, df ):
        self.df = df
        self.df = self.df.reset_index(drop=True)
        self.transform = NormalizePAD((1,64, 128))
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 

        file_name = self.df['file_name'][idx]
        #print(file_name)
        
        #file_name = file_name.replace('./train','/workspace/meta_trOCR/trocr/train')
        text = self.df['gt'][idx]
        #print(file_name)
        image = cv2.imread( file_name,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(128,64))
        
        #x = torch.tensor(np.array(image))
        
        
        skeleton_image = skeleton_text(image)

        edge_image = edge_text(image)  
        ink_dist = ink_local(image).astype(np.float32)
        

        image = self.transform(image)

        #image = torch.tensor(image).resize_(1,128,64).to(dtype=torch.float32)
        #edge_image = self.transform(edge_image)
        #skeleton_image = self.transform(skeleton_image)
        skeleton_image = torch.tensor(skeleton_image).resize_(1,128,64).to(dtype=torch.float32)
        #edge_image = torch.tensor(edge_image).resize_(1,128,64).to(dtype=torch.float32)
        ink_dist = torch.tensor(ink_dist).resize_(128).to(dtype=torch.float32)
        return image , skeleton_image , edge_image , ink_dist ,text 
    def edge_normalize(self,data):
        normalized_data = data  / 255
        return normalized_data

    def skeleton_normalize(self,data):
        normalized_data = data  / 255
        return normalized_data

class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


