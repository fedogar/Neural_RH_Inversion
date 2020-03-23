"""
Created on Fri Mar  6 14:50:03 2020

@author: ferrannew
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Charge_Repo.Charge import ChargerIN
from Charge_Repo.Charge import ChargerOUT

from Processing_Repo.Processing import LinearNet
from Processing_Repo.Processing import ModuleList
from Processing_Repo.Processing import LoadModule
from Processing_Repo.Processing import GANS
from Processing_Repo.Processing import GANS_TRAINING

from MDict_Repo.MDict import DicStruct
from MDict_Repo.MDict import DicUpload

if __name__ == "__main__":
    main()

def main():

    print('Select module for the traing form :' + ModuleList.list())
    Module=input()

    if Module =='LinearNet':
        Module = LinearNet
        Load = LoadModule( ChargerIN, ChargerOUT, Module )
        net= Load.module
        losses = Load.losses
        outputs = Load.outputs
        optimizers = Load.optimizers
    
    if Module =='GANS':
        Module = GANS
        Load =GANS_TRAINING(ChargerIn, ChargerOUT, GANS)

       


   

    