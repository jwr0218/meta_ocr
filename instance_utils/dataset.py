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
    

class OCRDataset(Dataset):
    def __init__(self, df ):
        self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['img_path'][idx]
        file_name = file_name.replace('./train','/workspace/meta_trOCR/trocr/train')
        text = self.df['text'][idx]
        #k_means = self.df['k_means_2'][idx]
        # prepare image (i.e. resize + normalize)
        image = cv2.imread( file_name,cv2.IMREAD_GRAYSCALE)
        #x = torch.tensor(np.array(image))
        
        
        # skeleton_image = skeleton_text(image)
        # edge_image = edge_text(image)
        # ink_dist = ink_local(image)

        image = torch.tensor(image).resize_(1,64,128).to(dtype=torch.float32)
        # skeleton_image = torch.tensor(skeleton_image).resize_(1,64,128).to(dtype=torch.float32)
        # edge_image = torch.tensor(edge_image).resize_(1,64,128).to(dtype=torch.float32)
        # ink_dist = torch.tensor(ink_dist).to(dtype=torch.float32)


        return image , text# ,skeleton_image,edge_image,ink_dist



class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.transform = transforms.Resize((imgH, imgW))
    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        # file_name , images ,labels , writer = zip(*batch)
        #images ,label , cluster = zip(*batch)
        images ,label = zip(*batch)
   
        labels = []
        for i in label:
            labels.append(i)


        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                image = image.squeeze(0)
                image = TF.to_pil_image(image)
                h,w = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)
                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        return image_tensors, np.array(labels) #,np.array(cluster)
    
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


