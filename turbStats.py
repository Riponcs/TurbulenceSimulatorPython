'''
This file contains the codes for generating correlation matrices for tilt
and higher-order abberations. 

Z. Mao, N. Chimitt, and S. H. Chan, "Accerlerating Atmospheric Turbulence 
Simulation via Learned Phase-to-Space Transform", ICCV 2021

Arxiv: https://arxiv.org/abs/2107.11627

Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
Copyright 2021
Purdue University, West Lafayette, IN, USA
'''

import math, os
import numpy as np
from scipy.special import jv
import scipy.integrate as integrate

def corr_mat(corr,save_path = './data'): 
    '''
    This function generates the correlation matrix for point spread functions (higher-order abberations)
    The correlation matrix will be stored in specified path with format 'R-corr_{}.npy'.format(corr)
    
    Input: 
        corr       -  Correlation strength. suggested range: (-5 ~ -0.01), with -5 has the
                      weakest correlation and -0.01 has the strongest. 
        save_path  -  Path to save the correlation matrix
    '''
    # corr: [-0.01, -0.1, -1, -5]
    num_zern=36
    N_rows=16
    N_cols=16

    subC = _nollCovMat(num_zern, 1, 1)
    C = np.zeros((int(N_rows*N_cols*num_zern), int(N_rows*N_cols*num_zern)))

    dist = np.zeros((N_rows,N_cols))
    for i in range(N_rows):
        for j in range(N_cols):
            for ii in range(N_rows):
                for jj in range(N_cols):
                    if not (i == ii and j == jj):
                        dist[ii,jj] = np.exp(corr*(np.linalg.norm(i - ii) + np.linalg.norm(j - jj)))
                    else:
                        dist[ii,jj] = 1
                    C[num_zern * (N_cols * i + j): num_zern * (N_cols * i + j + 1) \
                          , num_zern * (N_cols * ii + jj): num_zern * (N_cols * ii + jj + 1)] = dist[ii, jj] * subC

    e_val, e_vec = np.linalg.eig(C)

    R = np.real(e_vec * np.sqrt(e_val))
    np.save(os.path.join(save_path,'R-corr_{}.npy'.format(corr)),R)


def tilt_mat(N, D, r0, L, save_path = './data', thre = 0.002, adj = 1, use_temp = False): 
    '''
    This function generates the correlation matrix for tilt 
    The correlation matrix will be stored in specified path 
    with format 'S_half-size_{}-D_r0_{:.4f}.npy'.format(N,D_r0)
    
    Input: 
        N          -  Image size. suggested value: (128,256,512,1024)
        D          -  Apeture diameter
        r0         -  Fried parameter
        L          -  Propogation distance
        save_path  -  Path to save the correlation matrix
        thre       -  Used to suppress small valus in the correlation matrix. Increase 
                      this threshold if the pixel displacement appears to be scattering
        use_temp   -  If true, the correlation matrix will be stored in S_half-temp.npy. 
    '''
    # N: image size
    # D: Apeture diameter
    # r0: Fried parameter
    # L: Propagation distance
    D_r0 = D/r0
    wavelength = 0.500e-6
    k = 2*np.pi/wavelength
    delta0  = L*wavelength/(2*D)
    delta0 *= adj# Adjusting factor
    c1 = 2*((24/5)*math.gamma(6/5))**(5/6);
    c2 = 4*c1/np.pi*(math.gamma(11/6))**2;
    smax = delta0/D*N
    spacing = delta0/D
    I0_arr, I2_arr = _calculate_integral(smax, spacing)

    i, j = np.int32(N/2), np.int32(N/2)
    [x,y] = np.meshgrid(np.arange(1,N+0.01,1),np.arange(1,N+0.01,1))
    s = np.sqrt((x-i)**2 + (y-j)**2)
    s *= spacing

    C0 = (_In_m(s, spacing, I0_arr) + _In_m(s, spacing, I2_arr))/_I0(0)
    C0[i,j] = 1
    C0_scaled = C0*_I0(0)*c2*((D_r0)**(5/3))/(2**(5/3))*((2*wavelength/(np.pi*D))**2)*2*np.pi
    Cfft =  np.fft.fft2(C0_scaled)
    S_half =  np.sqrt(Cfft)
    S_half_max = np.max(np.max(np.abs(S_half)))
    S_half[np.abs(S_half) < thre*S_half_max] = 0
    S_half_new = np.zeros((2,N,N))
    S_half_new[0] = np.real(S_half)
    S_half_new[1] = np.imag(S_half)
    data = {}
    data['s_half'] = S_half_new
    data['const'] = np.sqrt(2)*N*(L/delta0)
    
    if use_temp: 
        np.save(os.path.join(save_path,'S_half-temp.npy'.format(N,D_r0)),data)
    else: 
        np.save(os.path.join(save_path,'S_half-size_{}-D_r0_{:.4f}.npy'.format(N,D_r0)),data)

        
def get_r0(Cn2, L, lbd=0.5e-6): 
    r0 = ((0.423 * (2*np.pi/lbd)**2) * Cn2 * integrate.quad(_f,0,L,args=L)[0]) ** (-3/5)
    
    return r0


def _f(z,L): 
    return (z/L)**(5/3)
        
        
def _nollToZernInd(j):
    """
    Authors: Tim van Werkhoven, Jason Saredy
    See: https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
    """
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))

    return n, m

def _nollCovMat(Z, D, fried):
    C = np.zeros((Z,Z))
    # Z: Number of Zernike Coeff's
    for i in range(Z):
        for j in range(Z):
            ni, mi = _nollToZernInd(i+1)
            nj, mj = _nollToZernInd(j+1)
            if (abs(mi) == abs(mj)) and (np.mod(i - j, 2) == 0):
                num = math.gamma(14.0/3.0) * math.gamma((ni + nj - 5.0/3.0)/2.0)
                den = math.gamma((-ni + nj + 17.0/3.0)/2.0) * math.gamma((ni - nj + 17.0/3.0)/2.0) * \
                      math.gamma((ni + nj + 23.0/3.0)/2.0)
                coef1 = 0.0072 * (np.pi ** (8.0/3.0)) * ((D/fried) ** (5.0/3.0)) * np.sqrt((ni + 1) * (nj + 1)) * \
                        ((-1) ** ((ni + nj - 2*abs(mi))/2.0))
                C[i, j] = coef1*num/den
            else:
                C[i, j] = 0
    C[0,0] = 1
    return C

def _I0(s):
    I0_s, _ = integrate.quad( lambda z: (z**(-14/3))*jv(0,2*s*z)*(jv(2,z)**2), 0, 1e3, limit = 100000)

    return I0_s

def _I2(s):
    I2_s, _ = integrate.quad( lambda z: (z**(-14/3))*jv(2,2*s*z)*(jv(2,z)**2), 0, 1e3, limit = 100000)
    
    return I2_s

def _calculate_integral(s_max, spacing):
    s_arr = np.arange(0,s_max,spacing)
    I0_arr = np.float32(s_arr*0)
    I2_arr = np.float32(s_arr*0)
    for i in range(len(s_arr)):
        I0_arr[i] = _I0(s_arr[i])
        I2_arr[i] = _I2(s_arr[i])
        
    return I0_arr, I2_arr

def _In_m(s, spacing, In_arr):
    idx = np.int32(np.floor(s.flatten()/spacing))
    M,N = np.shape(s)[0], np.shape(s)[1]
    In = np.reshape(np.take(In_arr, idx), [M,N])

    return In
