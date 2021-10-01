'''
Demo code for imaging through turbulence simulation

Z. Mao, N. Chimitt, and S. H. Chan, "Accerlerating Atmospheric Turbulence 
Simulation via Learned Phase-to-Space Transform", ICCV 2021

Arxiv: https://arxiv.org/abs/2107.11627

Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
Copyright 2021
Purdue University, West Lafayette, IN, USA


Ripon - I'm Using demo.py to generate the Turbulence Images using same strength.
  
'''

from torch import int8
from simulator import Simulator
from turbStats import tilt_mat, corr_mat
import matplotlib.pyplot as plt
import torch
import glob
import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm

# Select device.
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('CPU')


'''
The corr_mat function is used to generate spatial-temporal correlation matrix 
for point spread functions. It may take over 10 minutes to finish. However, 
for each correlation value, it only needs to be computed once and can be 
used for all D/r0 values. You can also download the pre-generated correlation
matrix from our website. 
https://engineering.purdue.edu/ChanGroup/project_turbulence.html
'''

Folder = 'G:/Research/Turbulence/RAFT/RAFT/datasets/Sintel-perfect_ratio/**/*.png'
Existing = 'Sintel-perfect_ratio'
NewFolder = 'Sintel-perfect_ratio-Tur'

strength = 5

# Load image, permute axis if color
# x = plt.imread('./images/color.png')
imgList = [a.replace('\\', '/') for a in glob.glob(Folder, recursive=True)]
x = plt.imread(imgList[0])

width = x.shape[1]
height = x.shape[0]

# Uncomment the following line to generate correlation matrix
#corr_mat(-0.1,'./data/')


# Generate correlation matrix for tilt. Do this once for each different turbulence parameter. 
tilt_mat(width, 0.1, 0.02, 3000)
print('Tilt Map generated')




print('Now Start processing each Imges')
for aimg in tqdm(imgList):
    x = plt.imread(aimg)
    
    #print(x.shape[0], width)
    
    if x.shape[0]!=width:
        x = cv2.resize(x, (height,width), interpolation=cv2.INTER_CUBIC)
    
    if len(x.shape) == 3: 
        x = x.transpose((2,0,1))
    x = torch.tensor(x, device = device, dtype=torch.float32)
    
    # Simulate
    simulator = Simulator(strength, width).to(device, dtype=torch.float32)
    
    out = simulator(x).detach().cpu().numpy()
    
    if len(out.shape) == 3: 
        out = out.transpose((1,2,0))
    
    out = np.clip(out, 0, 1)
    #print('\t\tChanged to = ',out.min(), out.max())    
    # save image
    NewFolderName = aimg.replace(Existing, NewFolder).rsplit('/', 1)[0]
    os.makedirs(NewFolderName, exist_ok=True)
    plt.imsave(aimg.replace(Existing, NewFolder), out)
    
    #plt.imsave(aimg.replace(Existing, NewFolder).replace('.jpeg', f'_{strength}_{width}.png'),out)


