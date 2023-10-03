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
        image = Image.open( file_name).convert("RGB")
        blurred_image = image.filter(ImageFilter.BLUR)
        #blurred_image = self.transform(image)

        x = torch.tensor(np.array(image)).resize_(1,64,128).to(dtype=torch.float32)
        blurred_x = torch.tensor(np.array(blurred_image)).resize_(1,64,128).to(dtype=torch.float32)
        return x ,blurred_x

#Parameters ========================

batch_size = 24
cluster_temperature =  1.0
instance_temperature =  0.5
learning_rate =  0.0003
epochs =  50
cluster_num = 10
#Parameters ========================


#data ========================
#df_total=pd.read_csv('/workspace/meta_trOCR/trocr/cluster_df.csv')
df_total = pd.read_csv('/workspace/korean_ocr_dataset/Training/ipynb/handwritten_korean_df.csv')

df = df_total.sample(frac=0.3,ignore_index=True)
del df['Unnamed: 0']
r , c = df.shape
print('TOTAL : ',r)

#df.reindex(range(len(df)))
#print(df)
#df = df_total[:30000]
#df = df_total
train_dataset = OCRDataset(df)
train_loader = DataLoader(train_dataset , batch_size = batch_size , num_workers = 4 , pin_memory = True, shuffle=True)

#data ========================

device = torch.device("cuda")

model = writer_cluster_model(cluster_num=cluster_num).to(device)

instanceLoss = InstanceLoss(batch_size, instance_temperature, device ).to(device)
clusterLoss = ClusterLoss(cluster_num, cluster_temperature, device ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

best_total_loss = 1000
best_cluster_loss = 1000
best_instance_loss = 1000
for i in range(epochs):
    cnt =0 
    for x_batch  in train_loader:

        
        image ,blurred_image = x_batch
        if image.shape[0] < batch_size:
            break

        image = image.to(device)
        blurred_image = blurred_image.to(device)
        

        latents = model(image,blurred_image)
        z_i, z_j, c_i, c_j = latents
        
        instance_loss = instanceLoss(z_i,z_j)
        current_instance_loss = instance_loss.item()
        cluster_loss = clusterLoss(c_i,c_j)
        current_cluster_loss = cluster_loss.item()
        total_loss = instance_loss + cluster_loss
        
        current_total_loss = total_loss.item()
        
        
        optimizer.zero_grad()
        total_loss.backward(retain_graph = True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        if cnt% 100 ==0 :
            print(f'{cnt*batch_size} / {r}\t\tInstance Loss : {instance_loss}\tCluster Loss : {cluster_loss}\tTotal Loss : {total_loss}')

            if current_total_loss < best_total_loss:
                best_total_loss = current_total_loss
                torch.save(model.state_dict(), f'saved_model_cluster/best_total_model_blur.pth')
            
            if current_cluster_loss < best_cluster_loss:
                best_cluster_loss = current_cluster_loss
                torch.save(model.state_dict(), f'saved_model_cluster/best_cluster_model_blur.pth')
            
            if current_instance_loss < best_instance_loss:
                best_instance_loss = current_instance_loss
                torch.save(model.state_dict(), f'saved_model_cluster/best_instance_model_blur.pth')        
        cnt +=1 


    print(f'Epoch {i}\nInstance Loss : {instance_loss}\tCluster Loss : {cluster_loss}\tTotal Loss : {total_loss}')
    



    