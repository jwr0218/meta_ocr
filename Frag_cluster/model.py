# from feature_extract.frag_cluster_Fragnet import FragNet
from feature_extract.FragNet_for_contrastive import FragNet
import torch
from Contrastive_Clustering.modules.network import Network_frag
from Contrastive_Clustering.modules.contrastive_loss import InstanceLoss , ClusterLoss
from PIL import Image
import pandas as pd 
import numpy as np 
from itertools import combinations
from torch.utils.data import Dataset , DataLoader
import torchvision.transforms as transforms

class writer_cluster_model(torch.nn.Module):
    def __init__(self,cluster_num):
        super().__init__()
        
        self.network = Network_frag(FragNet(1),512,256,cluster_num)
        
    def forward(self,image , blurred_image):

        z_i, z_j, c_i, c_j = self.network.forward(image,blurred_image)
        latent = [z_i, z_j, c_i, c_j]
        return latent
    
    def predict_cluseter(self,image ):

        z_i, c_i = self.network.forward_cluster(image)
        
        return z_i , c_i 