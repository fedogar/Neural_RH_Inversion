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

        if (checkpoint is None):
            files = glob.glob('trained/*.pth')
            self.checkpoint = max(files, key=os.path.getctime)
        else:
            self.checkpoint = '{0}'.format(checkpoint)
        
        self.model = model.Network(95*3+1, 100, 2).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        print("=> loading checkpoint '{}'".format(self.checkpoint))

        checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

        root = '/scratch1/aasensio/ferran/'
        print("Reading Enhanced_tau_530 - tau")
        tmp = io.readsav(f'{root}Enhanced_tau_530.save')
        self.T_tau = tmp['tempi'] #.reshape((86, 504*504))
        self.Pe_tau = tmp['epresi'] #.reshape((86, 504*504))
        self.tau = tmp['tau3'] / 5.0

        print("Reading Enhanced_tau_530 - z")
        tmp = io.readsav(f'{root}Enhanced_530_optiona_rh.save')
        self.T_z = tmp['tg'] #.reshape((161, 504*504))
        self.Pg_z = tmp['pg'] #.reshape((161, 504*504))
        self.z = tmp['z'] / 1e3

    def test(self):
        self.model.eval()
        index = 0
        top = 95

        T_z = self.T_z[0:top, 0:64, 0:64].reshape((top, 64*64)).T
        T_z = (T_z - 4000.0) / (15000.0 - 4000.0)
        Pg_z = np.log(self.Pg_z[0:top, 0:64, 0:64].reshape((top, 64*64)).T) / 10.0
        z = np.repeat(self.z[None, 0:top], 64*64, axis=0)

        ind_tau = 45
        tau = self.tau[ind_tau] * np.ones((64*64, 1))
        
        inp = np.concatenate([z, T_z, Pg_z, tau], axis=1)
        with torch.no_grad():

            inputs = torch.tensor(inp.astype('float32')).to(self.device)
            
            out = self.model(inputs)

        out = out.cpu().numpy()

        f, ax = pl.subplots(ncols=3, nrows=2, figsize=(10,6))
        im = ax[0,0].imshow(out[:, 0].reshape((64, 64)), cmap=pl.cm.viridis)
        pl.colorbar(im, ax=ax[0,0])
        im = ax[0,1].imshow(out[:, 1].reshape((64, 64)), cmap=pl.cm.viridis)
        pl.colorbar(im, ax=ax[0,1])
        im = ax[1,0].imshow((self.T_tau[ind_tau, 0:64, 0:64] - 4000.0)/ (15000.0 - 4000.0), cmap=pl.cm.viridis)
        pl.colorbar(im, ax=ax[1,0])
        im = ax[1,1].imshow(np.log(self.Pe_tau[ind_tau, 0:64, 0:64]) / 10.0, cmap=pl.cm.viridis)
        pl.colorbar(im, ax=ax[1,1])
        ax[0,2].plot(out[:, 0].reshape((64, 64)).flatten(), self.T_tau[ind_tau, 0:64, 0:64].flatten() / 1e3, '.')
        ax[1,2].plot(out[:, 1].reshape((64, 64)).flatten(), np.log(self.Pe_tau[ind_tau, 0:64, 0:64].flatten()) / 10.0, '.')        
        pl.show()
            
if (__name__ == '__main__'):
    
    deepnet = Testing(gpu=0, checkpoint=None)

    deepnet.test()