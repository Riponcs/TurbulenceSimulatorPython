'''
This file contains the codes for the P2S module and the simulator. Note that this file works for pytorch 1.4. use simulator.py for latest pytorch. 

Z. Mao, N. Chimitt, and S. H. Chan, "Accerlerating Atmospheric Turbulence 
Simulation via Learned Phase-to-Space Transform", ICCV 2021

Arxiv: https://arxiv.org/abs/2107.11627

Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
Copyright 2021
Purdue University, West Lafayette, IN, USA
'''

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Simulator(nn.Module): 
    '''
    Class variables:  
        Dr0       -  D/r0, which characterizes turbulence strength. suggested range: (1 ~ 5)
        img_size  -  Size of the image used for simulation. suggested value: (128,256,512,1024)
        corr      -  Correlation strength for PSF. suggested range: (-5 ~ -0.01)
        data_path -  Path where the model, PSF dictionary, and correlation matrices are stored
        device    -  'cuda:0' for GPU or 'CPU'
        scale     -  Used to artificially increase or decrase turbulence strength
        use_temp  -  If true, the correlation matrix for tilt will be loaded from S_half-temp.npy. 
    '''
    def __init__(self, Dr0, img_size, corr = -0.1, data_path = './data', device = 'cuda:0', scale=1.0, use_temp = False):
        super().__init__()
        self.img_size = img_size
        self.initial_grid = 16
        self.Dr0 = torch.tensor(Dr0)
        self.device = torch.device(device)
        self.Dr0 = torch.tensor(Dr0).to(self.device,dtype=torch.float32)
        self.mapping = _P2S()
        self.mapping.load_state_dict(torch.load(os.path.join(data_path,'P2S_model.pt')))
        self.dict_psf = np.load(os.path.join(data_path,'dictionary.npy'), allow_pickle = True)
        self.mu = torch.tensor(self.dict_psf.item()['mu']).reshape((1,1,33,33)).to(self.device,dtype=torch.float32)
        self.dict_psf = torch.tensor(self.dict_psf.item()['dictionary'][:100,:]).reshape((100,1,33,33))
        self.dict_psf = self.dict_psf.to(self.device,dtype=torch.float32)
        
        self.R = np.load(os.path.join(data_path,'R-corr_{}.npy'.format(corr)))
        self.R = torch.tensor(self.R).to(self.device,dtype=torch.float32)
        self.offset = torch.tensor([31,31]).to(self.device,dtype=torch.float32)
        
        if use_temp: 
            self.S_half = np.load(os.path.join(data_path,'S_half-temp.npy'.format(img_size,Dr0)), allow_pickle=True)
        else:
            self.S_half = np.load(os.path.join(data_path,'S_half-size_{}-D_r0_{:.4f}.npy'.format(img_size,Dr0)), allow_pickle=True)
        self.const = self.S_half.item()['const']
        self.S_half = torch.tensor(self.S_half.item()['s_half']).to(self.device,dtype=torch.float32)
        
        xx = torch.arange(0, img_size).view(1,-1).repeat(img_size,1)
        yy = torch.arange(0, img_size).view(-1,1).repeat(1,img_size)
        xx = xx.view(1,1,img_size,img_size).repeat(1,1,1,1)
        yy = yy.view(1,1,img_size,img_size).repeat(1,1,1,1)
        self.grid = torch.cat((xx,yy),1).permute(0,2,3,1).to(self.device,dtype=torch.float32)
        
        self.scale=scale


    def forward(self, img): 
        img_pad = F.pad(img.view((-1,1,self.img_size,self.img_size)), (16,16,16,16), mode = 'reflect')
        img_mean = F.conv2d(img_pad, self.mu).squeeze()
        dict_img = F.conv2d(img_pad, self.dict_psf)
        random_ = torch.sqrt(self.Dr0 ** (5 / 3))*torch.randn((self.initial_grid**2*36),1,device=self.device)

        zer = torch.matmul(self.R,random_).view(self.initial_grid,self.initial_grid,36).permute(2,0,1).unsqueeze(0)
        zer = F.interpolate(zer,size=(self.img_size,self.img_size),mode='bilinear', align_corners=False)
        zer = zer * self.scale
        weight = self.mapping(zer.squeeze().permute(1,2,0).view(self.img_size**2,-1))
        
        weight = weight.view((self.img_size,self.img_size,100)).permute(2,0,1)# target: (100,512,512)
        out = torch.sum(weight.unsqueeze(0)*dict_img,1) + img_mean
                
        MVx = torch.ifft((self.S_half*torch.randn(self.img_size,self.img_size,device=self.device)).permute(1,2,0),2)
        MVy = torch.ifft((self.S_half*torch.randn(self.img_size,self.img_size,device=self.device)).permute(1,2,0),2)
        pos = torch.stack((MVx[:,:,0],MVy[:,:,0]),2) * self.const
        flow = self.grid+pos
        flow = 2.0*flow / (self.img_size-1) - 1.0
        out = F.grid_sample(out.view((1,-1,self.img_size,self.img_size)), flow, 'bilinear', padding_mode='border', align_corners=False).squeeze()

        return out
    
class _P2S(nn.Module): 
    def __init__(self, input_dim = 36, output_dim = 100): 
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.out = nn.Linear(100, output_dim)

    def forward(self, x): 
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc2(y))
        out = self.out(y)

        return out


    
    
    

