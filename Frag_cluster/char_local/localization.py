from feature_extract.frag_cluster_Fragnet import FragNet
import torch
import torchvision.transforms as transforms
from Contrastive_Clustering.modules.network import Network_frag
from Contrastive_Clustering.modules.contrastive_loss import InstanceLoss , ClusterLoss
from PIL import Image , ImageFilter
import pandas as pd 
import numpy as np 
from itertools import combinations
from torch.utils.data import Dataset , DataLoader
#from model_5_MLP import writer_cluster_model
from model import writer_cluster_model
import math
class OCRDataset(Dataset):
    def __init__(self, df ):
        self.df = df
        self.transform = transforms.Compose([
            
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip()
        ])
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['img_path'][idx]
        file_name = file_name.replace('./train','/workspace/meta_trOCR/trocr/train')
        image = Image.open( file_name).convert("RGB")
        blurred_image = image.filter(ImageFilter.BLUR)
        #blurred_image = self.transform(image)

        x = torch.tensor(np.array(image)).resize_(1,64,128).to(dtype=torch.float32)
        blurred_x = torch.tensor(np.array(blurred_image)).resize_(1,64,128).to(dtype=torch.float32)
        return x ,blurred_x

#Parameters ========================

batch_size = 8
cluster_temperature =  1.0
instance_temperature =  0.5
learning_rate =  0.0003
epochs =  300
cluster_num = 15
#Parameters ========================


#data ========================
df_total=pd.read_csv('/workspace/meta_trOCR/trocr/cluster_df.csv')
df = df_total[:30000]
train_dataset = OCRDataset(df)
train_loader = DataLoader(train_dataset , batch_size = batch_size , num_workers = 4 , pin_memory = True, shuffle=True)
