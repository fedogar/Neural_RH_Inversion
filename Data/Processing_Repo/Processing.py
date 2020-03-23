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

class GANS_TRAINING( ChargerIn, ChargerOUT, GANS):

    def __init__():
        
        trainSpace = []
        
        for i in ChargerIN:
            discriminator = DiscriminatorNet()
            generator= GeneratorNet()
            trainSpace.append([discriminator,generator])
        
        self.trainS = trainSpace

    def train():
        l=0;
        for i in range(ChargerIN.in_())
            x = GANS.SammplingLoad(i)
            noise = GANS.noise(x.size()[0])

            



         



#Modules included
class LinearNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(161,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,86)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    
    def linearize (self,x): 
        x = torch.from_numpy(x)
        x = x.view(-1,161)
        return x

#Actually is passed linear net, but in main we can pass the model we want with correct match ChargeIN, ChargeOut 
class LoadModule (ChargerIN, ChargerOUT, LinearNet):
    
    def __init__ (self, ChargerIN, ChargerOUT):

        trainList=[]

        self.training = ChargerIN().__in
        self.testing = ChargerOUT().__out
        self.module = LinearNet()
        self.train(trainList)
        self.outputs = self.output(trainList)
        self.losses = self.loss(trainList)
        self.optimizers = self.optimize(trainList) 

    def train(self,trainList):

        l=0
        for i in self.training:
            i.view(-1,161)
            for j in np.arange(i.size()[0]):
                output = self.module.forward(i[j][:])
                self.module.zero_grad()
                optimizer = optim.SGD(self.module.parameters(), lr=0.01)
                optimizer.zero_grad()
                loss = nn.MSELoss(output,self.testing[j][:])
                loss.backward()
                optimizer.step()
            l=l+1
            trainList.append([output,optimizer,loss])

    def output(self,trainList):
        return trainList[:][0]

    def loss(self,trainList):
        return trainList[:][2]

    def optimize(self,trainList):
        return trainList[:][1]





                


            


