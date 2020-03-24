"""
Created on Fri Mar  6 14:50:03 2020

@author: ferrannew
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#GANS MODULE
import argparse
import os
import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import torchvision.utils as vutils

from IPython.display import HTML

#LoadMoulde Module
from Charge_Repo.Charge import ChargerIN
from Charge_Repo.Charge import ChargerOUT


class ModuleList():
    def __init__(self):
        self.list = ['LinearNet', 'GANS', 'Convolutional']



class GANS (nn.Module):

    def SammplingLoad(self,sample):
        return sample.view(-1, 161)

    def noise(self,size):
        n = torch.randn(size, 161)
        return n

    class DiscriminatorNet(nn.Module):

        def __init__(self):
            super(DiscriminatorNet, self).__init__()
            n_features = 161
            n_out = 1

            self.hidden0 = nn.Sequential( nn.Linear(n_features, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3))
            self.hidden1 = nn.Sequential( nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3))
            self.hidden2 = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3))
            self.out = nn.Sequential(torch.nn.Linear(256, n_out),torch.nn.Sigmoid())

        def forward(self, x):
            x = self.hidden0(x)
            x = self.hidden1(x)
            x = self.hidden2(x)
            x = self.out(x)
            return x
 
    class GeneratorNet(nn.Module):

        def __init__(self):
            super(GeneratorNet, self).__init__()
            n_features = 161
            n_out = 86
            self.hidden0 = nn.Sequential(nn.Linear(n_features, 256),nn.LeakyReLU(0.2))
            self.hidden1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))
            self.hidden2 = nn.Sequential(nn.Linear(512, 1024),nn.LeakyReLU(0.2))
            self.out = nn.Sequential(nn.Linear(1024, n_out),nn.LeakyReLU())

        def forward(self, x):
            x = self.hidden0(x)
            x = self.hidden1(x)
            x = self.hidden2(x)
            x = self.out(x)
            return x

#Modules included
class LinearNet(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(161*3,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,86*3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def linearize (self,x): 
        x = torch.from_numpy(x)
        x = x.view(-1,161)
        return x

#Actually is passed linear net, but in main we can pass the model we want with correct match ChargeIN, ChargeOut 
class LoadModule (ChargerIN,ChargerOUT, LinearNet):
    
    def __init__ (self):

        self.training = ChargerIN().in_
        self.testing = ChargerOUT().out_
        self.module = LinearNet()
        self.optimDict = list()
        self.netDict = list()
        self.losses = list()

    def train(self,trainList):

        for i in np.arange(3):
            self.training[i] = self.training[i].view(-1,161)
            self.testing[i] = self.testing[i].view(-1,161)

        self.netDict.clear()
        self.netDict.clear()

        for epoch in np.arange(504*504):

            packTest = torch.cat((self.testing[0][epoch][:], self.testing[1][epoch][:],self.testing[2][0][:]),1)
            packTrain = torch.cat((self.training[0][epoch][:], self.testing[1][epoch][:],self.training[2][0][:]),1)
            
            output = self.module.forward(packTrain)
            self.module.zero_grad()
            optimizer = optim.SGD(self.module.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss = nn.MSELoss(output,packTest)
            self.losses.append(loss.backward())
            optimizer.step()

        self.optimDict.append(optimizer.state_dict)
        self.netDict.append(self.module.state_dict)

    def netState(self,trainList):
        return trainList[:][1]

    def optimState(self,trainList):
        return trainList[:][0]


class Convolutional1DNet(nn.Module):
    
    #Esta codeado para una longitud de entrada de 161 a 86: sea 86 la longitud de salida del sistema.
    def __init__():
        super().__init__():
        self.conv1 = nn.Conv1d(3,2,74)
        self.conv1 = nn.Conv1d(2,3,86)
    
    def forward(x):
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        return x

class LoadModuleConv (ChargerIN,ChargerOUT,Convolutional1DNet ):
    def __init__():
        self.training = ChargerIN().in_
        self.testing = ChargerOUT().in_
        self.module = Convolutional1DNet()
        self.oDictpxp = list()
        self.oDtpack = list()
        self.oNpxp = list()
        self.oNpxpack = list()
        self.losspack = list()
        self.losspxp = list()

    def testing_PxP():
        
        for i in np.arange(3):
            self.training[i] = self.training[i].view(-1,1,161)
            self.testing[i] = self.testing[i].view(-1,1,161)
            
        for epoch in np.arange(504*504):
            
            #In this form take convolution pixel to pixel / does not allow correlations between pixels ::
            packTest = torch.cat((self.testing[0][epoch][:], self.testing[1][epoch][:],self.testing[2][0][:]),1)
            packTrain = torch.cat((self.training[0][epoch][:], self.testing[1][epoch][:],self.training[2][0][:]),1)
        
            output = self.module.forward(packTrain)
            self.module.zero_grad()
            optimizer = optim.SGD(self.module.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss = nn.MSELoss(output,packTest)
            self.losspxp.append(loss.backward())
            optimizer.step()
        
        self.oDpxp.append(optimizer.state_dict)
        self.nDpxp.append(self.module.state_dict)


    def testing_PackxPack():

        for i in np.arange(3):
            self.training[i] = self.training[i].view(-1,1,161)
            self.testing[i] = self.testing[i].view(-1,1,161)
            
        for epoch in np.arange(3969):
            
            #In this form take convolution pixel to pixel / does not allow correlations between pixels ::
            packTest = torch.cat((self.testing[0][epoch][64*epoch:64*(epoch+1)], self.testing[1][64*epoch:64*(epoch+1)][:],(self.testing[2][0][:].repeat(64,161)),1)
            packTrain = torch.cat((self.training[0][epoch][64*epoch:64*(epoch+1)], self.testing[1][epoch][64*epoch:64*(epoch+1)],(self.testing[2][0][:].repeat(64,161)),1)
        
            output = self.module.forward(packTrain)
            self.module.zero_grad()
            optimizer = optim.SGD(self.module.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss = nn.MSELoss(output,packTest)
            self.losspack.append(loss.backward())
            optimizer.step()
        
        self.oDpack.append(optimizer.state_dict)
        self.nDpack.append(self.module.state_dict)





                


            



