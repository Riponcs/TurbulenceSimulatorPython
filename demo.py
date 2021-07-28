'''
Demo code for imaging through turbulence simulation

Z. Mao, N. Chimitt, and S. H. Chan, "Accerlerating Atmospheric Turbulence 
Simulation via Learned Phase-to-Space Transform", ICCV 2021

Arxiv: https://arxiv.org/abs/2107.11627

Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
Copyright 2021
Purdue University, West Lafayette, IN, USA
'''

from simulator import Simulator
from turbStats import tilt_mat, corr_mat
import matplotlib.pyplot as plt
import torch

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

# Uncomment the following line to generate correlation matrix
# corr_mat(-0.1,'./data/')

# Generate correlation matrix for tilt. Do this once for each different turbulence parameter. 
tilt_mat(x.shape[1], 0.1, 0.05, 3000)

# Load image, permute axis if color
x = plt.imread('./images/color.png')
if len(x.shape) == 3: 
    x = x.transpose((2,0,1))
x = torch.tensor(x, device = device, dtype=torch.float32)

# Simulate
simulator = Simulator(2, 512).to(device, dtype=torch.float32)

out = simulator(x).detach().cpu().numpy()

if len(out.shape) == 3: 
    out = out.transpose((1,2,0))

# save image
plt.imsave('./images/out.png',out)


