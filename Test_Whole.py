#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:01:34 2020

@author: ferran_2020
"""


import shutil
import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.functional as tf
import torch.utils.data
import time
from tqdm import tqdm
import model
import glob
import os
import scipy.io as io

class Testing(object):
    def __init__(self, gpu=0, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        
        self.model = model.Network(161*3, 100, 86*3).to(self.device)
        
        root = '/Users/ferran_2020/TFG/Neural_RH_Inversion/'
        print("Reading Enhanced_tau_530 - tau")
        tmp = io.readsav(f'{root}Enhanced_tau_530.save')
        self.T_tau = tmp['tempi'] .reshape((86, 504*504))
        self.Pe_tau = tmp['epresi'] .reshape((86, 504*504))

        print("Reading Enhanced_tau_530 - z")
        tmp = io.readsav(f'{root}Enhanced_530_optiona_rh.save')
        self.T_z = tmp['tg'] #.reshape((161, 504*504))
        self.Pg_z = tmp['pg'] #.reshape((161, 504*504))
        self.z = tmp['z'] / 1e3
        
        import pickle
        filename = '/Users/ferran_2020/TFG/Neural_RH_Inversion/checkpoint_Whole.dict'
    
        with open (filename , 'rb') as f:
            s = pickle.load(f)
        f.close()
        self.model.load_state_dict(s[0])


    def test(self):
        
        import copy
        import numpy as np
        
        self.model.eval()
        index = 0
        top = 161

        root = '/Users/ferran_2020/TFG/Neural_RH_Inversion/'
    
        
        T_z = self.T_z[0:top, 0:504, 0:504].reshape((top, 504*504))
        T_z = (T_z - 4000.0) / (15000.0 - 4000.0)
        Pg_z = np.log(self.Pg_z[0:top, 0:504, 0:504].reshape((top, 504*504))) / 10.0
        
        tmp = io.readsav(f'{root}Enhanced_tau_530.save')
        self.T_tau = tmp['tempi'].reshape((86, 504,504))
        self.Pe_tau = tmp['epresi'].reshape((86, 504,504))
        T_tau = (self.T_tau - 4000.0) / (15000.0 - 4000.0)
        Pe_tau = np.log(self.Pe_tau) / 10.0
        T_tau = np.array(T_tau)
        Pe_tau = np.array(Pe_tau)

        
        
        
        X = torch.randn((504*504,3,86))
        
        for i in np.arange(504*504):
        
            inp = np.hstack([self.z, T_z[:,i], Pg_z[:,i]])     
        
            with torch.no_grad():

                inputs= torch.tensor(inp.astype('float32')).to(self.device)
            
                out = self.model(inputs)
                out = out.reshape(3,86)
                
                X[i] = copy.deepcopy(out)
         
        X = X.view(504,504,3,86)
        
        Z,T,P = torch.chunk(X,3, dim = 2)
        P = P.squeeze();T = T.squeeze();
        
        MacroT = np.ones((504,504))
        MacroP = np.ones((504,504))
       
        MacroTM = np.ones((504,504))
        MacroPM = np.ones((504,504))
        
        MacroTm = np.ones((504,504))
        MacroPm = np.ones((504,504))
        
        MacroTs = np.ones((504,504))
        MacroPs = np.ones((504,504))
      
        for i in np.arange(504):
            for j in np.arange(504):
             
                MacroTM[i][j] = torch.max(T[i][j]).detach().numpy()
                MacroPM[i][j] = torch.max(P[i][j]).detach().numpy()
                MacroT[i][j] = torch.median(T[i][j]).detach().numpy()
                MacroP[i][j] = torch.median(P[i][j]).detach().numpy()
                MacroTs[i][j] = torch.mean(T[i][j]).detach().numpy()
                MacroPs[i][j] = torch.mean(P[i][j]).detach().numpy()
                MacroTm[i][j] = torch.min(T[i][j]).detach().numpy()
                MacroPm[i][j] = torch.min(P[i][j]).detach().numpy()
           
        
        import matplotlib.pyplot as plt
        
        fig, ax  = plt.subplots(2,2)
        im1 = ax[0][0].imshow(MacroP,aspect='equal' )
        ax[0][0].set_title('Median Pressure')
        pl.colorbar(im1, ax=ax[0,0])  
        im2 = ax[1][1].imshow(MacroPM,aspect='equal' )
        ax[1][1].set_title('Max Pressure')
        pl.colorbar(im2, ax=ax[1,1])  
        im3 = ax[1][0].imshow(MacroPm , aspect='equal' )
        ax[1][0].set_title('Min Pressure')
        pl.colorbar(im3, ax=ax[1,0])  
        im4 = ax[0][1].imshow(MacroPs,aspect='equal' )
        ax[0][1].set_title('Mean Pressure')
        pl.colorbar(im4, ax=ax[0,1])     
        
        fig, ax  = plt.subplots(2,2)
        im1 = ax[0][0].imshow(MacroT,aspect='equal' )
        ax[0][0].set_title('Median Temperature')
        pl.colorbar(im1, ax=ax[0,0])
        im2 = ax[1][1].imshow(MacroTM,aspect='equal' )
        ax[1][1].set_title('Max Temperature')
        pl.colorbar(im2, ax=ax[1,1])
        im3 = ax[1][0].imshow(MacroTm,aspect='equal' )
        ax[1][0].set_title('Min Temperature')
        pl.colorbar(im3, ax=ax[1,0])
        im4 = ax[0][1].imshow(MacroTs,aspect='equal' )
        ax[0][1].set_title('Mean Temperature')
        pl.colorbar(im4, ax=ax[0,1])    
   
    
        fig, ax  = plt.subplots(2,2)
        im1 = ax[0][0].imshow(T.detach().numpy()[:,:,85],aspect='equal' )
        ax[0][0].set_title('Temperatura Reconstuida')
        pl.colorbar(im1, ax=ax[0,0])
        im2 = ax[1][0].imshow(T_tau[85][:][:],aspect='equal' )
        ax[1][0].set_title('Temperatura Original')
        pl.colorbar(im2, ax=ax[1,1])
        im3 = ax[1][1].imshow(Pe_tau[85][:][:],aspect='equal' )
        ax[1][1].set_title('Presión Original')
        pl.colorbar(im3, ax=ax[1,0])
        im4 = ax[0][1].imshow(P.detach().numpy()[:,:,85],aspect='equal' )
        ax[0][1].set_title('Presión Reconstruida')
        pl.colorbar(im4, ax=ax[0,1])
        
        
        fig, ax  = plt.subplots(2,2)
        im1 = ax[0][0].imshow(T.detach().numpy()[:,:,40],aspect='equal' )
        ax[0][0].set_title('Temperatura Reconstuida')
        pl.colorbar(im1, ax=ax[0,0])
        im2 = ax[1][0].imshow(T_tau[40][:][:],aspect='equal' )
        ax[1][0].set_title('Temperatura Original')
        pl.colorbar(im2, ax=ax[1,1])
        im3 = ax[1][1].imshow(Pe_tau[40][:][:],aspect='equal' )
        ax[1][1].set_title('Presión Original')
        pl.colorbar(im3, ax=ax[1,0])
        im4 = ax[0][1].imshow(P.detach().numpy()[:,:,40],aspect='equal' )
        ax[0][1].set_title('Presión Reconstruida')
        pl.colorbar(im4, ax=ax[0,1])
        
        
        fig, ax  = plt.subplots(2,2)
        im1 = ax[0][0].imshow(T.detach().numpy()[:,:,2],aspect='equal' )
        ax[0][0].set_title('Temperatura Reconstuida')
        pl.colorbar(im1, ax=ax[0,0])
        im2 = ax[1][0].imshow(T_tau[2][:][:],aspect='equal' )
        ax[1][0].set_title('Temperatura Original')
        pl.colorbar(im2, ax=ax[1,1])
        im3 = ax[1][1].imshow(Pe_tau[2][:][:],aspect='equal' )
        ax[1][1].set_title('Presión Original')
        pl.colorbar(im3, ax=ax[1,0])
        im4 = ax[0][1].imshow(P.detach().numpy()[:,:,2],aspect='equal' )
        ax[0][1].set_title('Presión Reconstruida')
        pl.colorbar(im4, ax=ax[0,1])
    
   
         
deepnet_Test_Whole = Testing(gpu=0, checkpoint=None)
deepnet_Test_Whole.test()
