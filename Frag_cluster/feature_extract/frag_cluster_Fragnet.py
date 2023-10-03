#*************************
# Deep Learning Package for Writer Identification
# @author: Sheng He
# @Email: heshengxgd@gmail.com
# Github: https://github.com/shengfly/writer-identification


# Sheng He, Lambert Schomaker,  FragNet: Writer Identification Using Deep Fragment Networks,
# IEEE Transactions on Information Forensics and Security ( Volume: 15), Pages: 3013-3022
# @Arixv: https://arxiv.org/pdf/2003.07212.pdf

#*************************


import torch
import torch.nn as nn

class VGGnet(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
        
        layers=[64,128,256,512]
        
        self.conv1 = self._conv(input_channel,layers[0])
        self.maxp1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = self._conv(layers[0],layers[1])
        self.maxp2 = nn.MaxPool2d(2,stride=2)
        self.conv3 = self._conv(layers[1],layers[2])
        self.maxp3 = nn.MaxPool2d(2,stride=2)
        self.conv4 = self._conv(layers[2],layers[3])
        self.maxp4 = nn.MaxPool2d(2,stride=2)
        
        
    def _conv(self,inplance,outplance,nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance,outplance,kernel_size=3,
                          stride=1,padding=1,bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance
            
        conv = nn.Sequential(*conv)
               
        return conv
    
    def forward(self, x):
        xlist=[x]
        x = self.conv1(x)
        xlist.append(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        xlist.append(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        xlist.append(x)
        x = self.maxp3(x)
        x = self.conv4(x)
        xlist.append(x)
        return xlist


        
class FragNet(nn.Module):
    def __init__(self,inplace):
        super().__init__()
        
        self.net = VGGnet(inplace)
        
        layers=[64,128,256,512,512]
        
        self.conv0 = self._conv(inplace,layers[0])
        self.conv1 = self._conv(layers[0]*2,layers[1])
        self.maxp1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = self._conv(layers[1]*2,layers[2])
        self.maxp2 = nn.MaxPool2d(2,stride=2)
        self.conv3 = self._conv(layers[2]*2,layers[3])
        self.maxp3 = nn.MaxPool2d(2,stride=2)
        self.conv4 = self._conv(layers[3]*2,layers[4])
        self.maxp4 = nn.MaxPool2d(2,stride=2)
        
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.MLP0 = nn.Sequential(
        nn.Linear(64*64*64, 512),
        nn.ReLU(),
        nn.Linear(512, 1024)
        )
        self.MLP1 = nn.Sequential(
        nn.Linear(128*32*32, 512),
        nn.ReLU(),
        nn.Linear(512, 1024)
        )

        self.MLP2 = nn.Sequential(
        nn.Linear(256*16*16, 512),
        nn.ReLU(),
        nn.Linear(512, 1024)
        )
        self.MLP3 = nn.Sequential(
        nn.Linear(512*8*8, 512),
        nn.ReLU(),
        nn.Linear(512, 1024)
        )
        self.MLP4 = nn.Sequential(
        nn.Linear(512*4*4, 512),
        nn.ReLU(),
        nn.Linear(512, 1024)
        )

        #noise = torch.randn(tensor.shape) << 노이즈 주는 코드 
        #noisy_tensor = tensor + noise


        
    def _conv(self,inplance,outplance,nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance,outplance,kernel_size=3,
                          stride=1,padding=1,bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance
            
        conv = nn.Sequential(*conv)
               
        return conv
    
    def forward(self,x):
        xlist = self.net(x)
        step = 16
        feature_list = []
        # input image
        reslist = []
        fratten_0 = []
        for n in range(0,65,step):
            xpatch = xlist[0][:,:,:,n:n+64]
            r = self.conv0(xpatch)
            
            num_features = r.size(1) * r.size(2) * r.size(3)
            r_f = r.view(r.size(0), num_features)
            r_f = self.MLP0(r_f)
            fratten_0.append(r_f)
            reslist.append(r)
        fratten_0 = torch.stack(fratten_0)
        fratten_0 = fratten_0.permute(1,0,2)
        fratten_0 = fratten_0.flatten(start_dim=1, end_dim=2)
        # 0-layer
        
        
        
        idx = 0
        res1list = []
        fratten_1 = []
        for n in range(0,65,step):
            xpatch = xlist[1][:,:,:,n:n+64]
            xpatch = torch.cat([xpatch,reslist[idx]],1)
            idx += 1
            r = self.conv1(xpatch)
            r = self.maxp1(r)
            
            num_features = r.size(1) * r.size(2) * r.size(3)
            r_f = r.view(r.size(0), num_features)
            r_f = self.MLP1(r_f)

            fratten_1.append(r_f)

            res1list.append(r)
        
        fratten_1 = torch.stack(fratten_1)
        fratten_1 = fratten_1.permute(1,0,2)
        fratten_1 = fratten_1.flatten(start_dim=1, end_dim=2)

        # 1-layer
        idx = 0
        res2list = []
        step = 8
        fratten_2 = []
        for n in range(0,33,step):
            xpatch = xlist[2][:,:,:,n:n+32]
            xpatch = torch.cat([xpatch,res1list[idx]],1)
            idx += 1
            r = self.conv2(xpatch)
            r = self.maxp2(r)
            num_features = r.size(1) * r.size(2) * r.size(3)
            r_f = r.view(r.size(0), num_features)
            r_f = self.MLP2(r_f)
            fratten_2.append(r_f)
            
            
            res2list.append(r)
        fratten_2 = torch.stack(fratten_2)
        fratten_2 = fratten_2.permute(1,0,2)
        fratten_2 = fratten_2.flatten(start_dim=1, end_dim=2)
        # 2-layer
        
        idx = 0
        res3list = []
        step = 4
        fratten_3 = []
        for n in range(0,17,step):
            xpatch = xlist[3][:,:,:,n:n+16]
            xpatch = torch.cat([xpatch,res2list[idx]],1)
            idx += 1
            r = self.conv3(xpatch)
            r = self.maxp3(r)
            num_features = r.size(1) * r.size(2) * r.size(3)
            r_f = r.view(r.size(0), num_features)
            r_f = self.MLP3(r_f)
            fratten_3.append(r_f)
            res3list.append(r)
        fratten_3 = torch.stack(fratten_3)
        fratten_3 = fratten_3.permute(1,0,2)
        fratten_3 = fratten_3.flatten(start_dim=1, end_dim=2)
        # 3-layer
        idx = 0
        step = 2
        res4list = []
        fratten_4 = []
        for n in range(0,9,step):
            xpatch = xlist[4][:,:,:,n:n+8]
            xpatch = torch.cat([xpatch,res3list[idx]],1)
            idx += 1
            r = self.conv4(xpatch)
            r = self.maxp4(r)
            num_features = r.size(1) * r.size(2) * r.size(3)
            r_f = r.view(r.size(0), num_features)
            r_f = self.MLP4(r_f)
            fratten_4.append(r_f)

            res4list.append(r)
            
        fratten_4 = torch.stack(fratten_4)
        fratten_4 = fratten_4.permute(1,0,2)
        fratten_4 = fratten_4.flatten(start_dim=1, end_dim=2)

        feature_list = [reslist,res1list,res2list,res3list,res4list]
        fratten = [fratten_1,fratten_2,fratten_3,fratten_4]
        return feature_list ,fratten
        

if __name__ == '__main__':
    x = torch.rand(4,1,64,128)
    #batch / 1 , 62 / 128 
    mod = FragNet(1,105)

    feature_list ,fratten = mod(x)
    for y in feature_list:
        print('-'*20)
        print(f'length = {len(y)}')
        for feature in y:
            print(feature.shape)
    print('===='*20,'fratten','===='*20)

    for y in fratten:

        print(f'shape = {y.shape}')
        
    


