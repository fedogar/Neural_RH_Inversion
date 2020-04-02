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


class ModuleList(object):
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
        super(LinearNet,self).__init__()
        self.fc1 = nn.Linear(161*3,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,86*3)

    def forward(self,x,y,z):
        x = F.threshold(self.fc1(x),y,z)
        x = F.threshold(self.fc2(x),y,z)
        x = F.threshold(self.fc3(x),y,z)
        x = self.fc4(x)
        return x

    def linearize (self,x): 
        x = torch.from_numpy(x)
        x = x.view(-1,161)
        return x

#Actually is passed linear net, but in main we can pass the model we want with correct match ChargeIN, ChargeOut 
class LoadModule (object):
    
    def __init__ (self,ChargerIN,ChargerOUT,LinearNet):
        super(LoadModule,self).__init__()
        self.lr = 0.05
        self.training = ChargerIN().in_
        self.testing = ChargerOUT().out_
        self.module = LinearNet()
        self.netDict = {}
        self.losses = list()

    def train (self, lr_):

        for i in np.arange(3):
            self.training[i] = self.training[i].view(-1,161)
            self.testing[i] = self.testing[i].view(-1,161)

        self.netDict.clear()
        self.optimDict.clear()

        for epoch in np.arange(504*504):

            packTest = torch.cat((self.testing[0][epoch][:], self.testing[1][epoch][:],self.testing[2][0][:]),1)
            packTrain = torch.cat((self.training[0][epoch][:], self.testing[1][epoch][:],self.training[2][0][:]),1)
            
            output = self.module.forward(packTrain.float())
            self.module.zero_grad()
            optimizer = optim.SGD(self.module.parameters(), lr_)
            optimizer.zero_grad()
            loss =F.mse_loss(output,packTest)
            if loss == float('Inf'):
                break 
            loss.backward()
            self.losses.append(loss)
            optimizer.step()
            try:
                self.netDict = { 'fc1' : self.fc1.weight, 'fc2' : self.fc2.weight, 'fc3' :self.fc3.weight , 'fc4' : self.fc4.weight } 
            except NameError:
                    print('Error')

class Convolutional1DNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(3,2,74)
        self.conv2 = nn.Conv1d(2,3,8,padding=10 )
        self.conv3 = nn.Conv1d(3,2,14)
        self.conv4 = nn.Conv1d(2,3,3)
        
        
    def forward(self,x):
        
        
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = self.conv4(x)
        
        return x
    
#It needs to be passed this 3 modules in the terminal call one is load
class LoadModuleCL (object):
    def __init__(self,ChargerIN,ChargerOUT,Convolutional1DNet):
        self.lr = 0.09
        self.module2 = Convolutional1DNet()
        self.training = ChargerIN().in_
        self.testing = ChargerOUT().out_
        self.nDpxp = {}
        self.nDpack = {}
        self.losspack = list()
        self.losspxp = list()
    def testing_PackxPack(self,lr_):
        self.lr = lr_
        for i in np.arange(3):
            self.training[i] = self.training[i].view(-1,1,161)
            self.testing[i] = self.testing[i].view(-1,1,86)
        for epoch in np.arange(3969):
            #In this form take convolution pixel to pixel / does not allow correlations between pixels ::
            packTest = torch.cat((self.testing[0][64*epoch:64*(epoch+1)][:], self.testing[1][64*epoch:64*(epoch+1)][:],(self.testing[2]).repeat(64,1,1)),1)
            packTrain = torch.cat((self.training[0][64*epoch:64*(epoch+1)][:], self.training[1][64*epoch:64*(epoch+1)][:],(self.training[2]).repeat(64,1,1)),1)
            self.module2 = (self.module2).float()
            output = (self.module2).forward(packTrain.float())
            (self.module2).zero_grad()
            optimizer = optim.SGD((self.module2).parameters(), lr_)
            optimizer.zero_grad()
            loss = F.mse_loss(output,packTest.float())
            if loss == float('Inf'):
                break 
            loss.backward()
            (self.losspack).append(loss)
            optimizer.step()
            if epoch == 3968:
                self.nDpack = {}

    def testing_PxP(self,lr_):
        self.lr = lr_
        for i in np.arange(3):
            self.training[i] = self.training[i].view(-1,1,161)
            self.testing[i] = self.testing[i].view(-1,1,86)
        for epoch in np.arange(504*504):
            #In this form take convolution pixel to pixel / does not allow correlations between pixels ::
            packTest = torch.cat((self.testing[0][epoch].view(-1,1,86), self.testing[1][epoch].view(-1,1,86),self.testing[2]),1)
            packTrain = torch.cat((self.training[0][epoch].view(-1,1,161), self.training[1][epoch].view(-1,1,161),self.training[2]),1)
            self.module2 = (self.module2).float()
            output = self.module2.forward(packTrain.float())
            self.module2.zero_grad()
            optimizer = optim.SGD(self.module2.parameters(), lr=lr_)
            optimizer.zero_grad()
            loss = F.mse_loss(output,packTest.float())
            if loss == float('Inf'):
                break 
            loss.backward()
            self.losspxp.append(loss)
            optimizer.step()
            if epoch == 254015:
                try:
                    self.nDpxp = { 'conv1' : self.module2.conv1.weight, 'conv2' : self.module2.conv2.weight, 'conv3' :self.module2.conv3.weight , 'conv4' : self.module2.conv1.weight } 
                except NameError:
                    self.nDpxp={ 'conv1' : self.module2.conv1.weight}
                                      
class LoadShuffle (object):
    
    def __init__(self,ChargerIN,ChargerOUT,Convolutional1DNet,ShuffleSplit):
        self.training = ChargerIN().in_
        self.testing = ChargerOUT().out_
        self.shuffel = self.prepare()
        self.randoomstate = 4
        
        def prepare(self):
            for i in no.arange(3):
                self.testing[i].view(-1,1,86)     
                self.training[i](-1,1,161)
            
            pack = torch.cat((self.testing[0],self.testing[1],self.testing[2].repeat(self.testing[0].size()[0],1,1),self.training[0],self.training[1],self.training[2].repeat(self.training[0].size()[0],1,1)),1)
            ss = ShuffleSplit(pack.size()[0], test_size=0.25, random_state=4)
            
            #Here try to shuffle the vector for the training          
"""
Here we define the classes related to the overfitting, underfitting checking for the different examples
"""
from sklearn.model_selection import ShuffleSplit

def ValidationTest (object):
    
    ss = ShuffleSplit(n_splits=2540156, test_size=0.25, random_state=0)
    cl = LoadModuleCL(ChargerIN,ChargerOUT,Convolutional1DNet,LinearNet)
    for i in no.arange(3):
        cl.testing[i].view(-1,1,86)     
        cl.training[i](-1,1,161)
    
    
    
    



            



