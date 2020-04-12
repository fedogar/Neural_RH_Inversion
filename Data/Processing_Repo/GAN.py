from __future__ import print_function

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML



#We convert the dataset to a image like, from vectors for the T and P rows, columns
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
        self.fc3 = nn.Linear(100,200)
        self.fc4 = nn.Linear(200,86*3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
