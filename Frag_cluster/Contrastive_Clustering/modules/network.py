import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c


class Network_frag(nn.Module):
    def __init__(self,model,input_dim ,feature_dim, class_num):
        super(Network_frag, self).__init__()
        self.input_dim =input_dim
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.input_dim , self.input_dim ),
            nn.ReLU(),
            nn.Linear(self.input_dim , self.feature_dim),
        )
        self.cluster_projector = nn.Linear(512,self.cluster_num)
        self.softmax_final = nn.Softmax(dim=1)
        self.model = model

    def forward(self, x, x_processed):
        h_i = self.model(x)
        h_j = self.model(x_processed)
        z_i_lst = []
        for h in h_i:
            z_i_lst.append(self.instance_projector(h))
        z_i = 0 
        for z in z_i_lst:
            z_i +=z
        
        z_j_lst = []
        for h in h_j:
            z_j_lst.append(self.instance_projector(h))
        z_j = 0 
        for z in z_j_lst:
            z_j +=z
        #======================================== z / c  ===================================
        c_i = 0 
        for h in h_i:
            c_i += self.cluster_projector(h)
        c_i = self.softmax_final(c_i)
        c_j = 0 
        for h in h_j:
            c_j += self.cluster_projector(h)
        c_j = self.softmax_final(c_j)
        return z_i, z_j, c_i, c_j
    
    def forward_cluster(self, x):
        h_i = self.model(x)
        z_i_lst = []
        for h in h_i:
            z_i_lst.append(self.instance_projector(h))
        z_i = 0 
        for z in z_i_lst:
            z_i +=z
        
        #======================================== z / c  ===================================
        c_i = 0 
        for h in h_i:
            c_i += self.cluster_projector(h)
        c_i = self.softmax_final(c_i)
        return z_i, c_i