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
        file_name = self.df['file_name'][idx]
        #file_name = file_name.replace('./train','/workspace/meta_trOCR/trocr/train')
        text = self.df['gt'][idx]
        image = Image.open( file_name).convert("RGB")
       
        #blurred_image = self.transform(image)

        x = torch.tensor(np.array(image)).resize_(1,64,128).to(dtype=torch.float32)
        return x ,file_name , text

#Parameters ========================

batch_size = 32
cluster_temperature =  1.0
instance_temperature =  0.5
learning_rate =  0.0003
epochs =  200
cluster_num = 10
#Parameters ========================


#data ========================
df_total = pd.read_csv('/workspace/korean_ocr_dataset/Training/ipynb/handwritten_korean_df.csv')
r , c = df_total.shape
df = df_total
test_dataset = OCRDataset(df)
test_loader = DataLoader(test_dataset , batch_size = batch_size , num_workers = 4 , pin_memory = True )

#data ========================

device = torch.device("cuda")

model = writer_cluster_model(cluster_num=cluster_num).to(device)
model_path = '/workspace/meta_trOCR/Frag_cluster/saved_model_cluster/best_total_model_blur1.pth'
model.load_state_dict(torch.load(model_path))

model.eval()
df = pd.DataFrame()
cnt = 0 
for x_batch  in test_loader:

    cnt+=1
    if cnt %100 == 0 :
        print(f'{cnt*batch_size}/\t{r}')
    image , file_name  ,text= x_batch
    #print(file_name,text)
    image = image.to(device)
    
    z_i, c_i  = model.predict_cluseter(image)

    argmax_indices = torch.argmax(c_i, dim=1)
    tmp_df = pd.DataFrame({'cluster': argmax_indices.cpu().numpy(), 'file_name': file_name , 'gt' : text})
    df = pd.concat([df,tmp_df],axis = 0,ignore_index=True)

    df.to_csv('korean_contrastive_cluster_result_10.csv')        
    