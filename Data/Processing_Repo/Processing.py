"""
Created on Fri Mar  6 14:50:03 2020

@author: ferrannew
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    
    def linearize (x): 
        x = torch.from_numpy(x)
        x = x.view(-1,161)
        return x

class LoadModule (ChargerIN, ChargerOUT, Module):

    trainList=[]

    def __init__ (self, ChargerIN, ChargerOUT):

        self.training = ChargerIN().__in
        self.testing = ChargerOUT().__out
        self.module = Module()
        self.train()
        self.outputs = output()
        self.losses = loss()
        self.optimizers = optimize() 

    def train(self):

        l=0
        for i in self.training
            i.view(-1,161)
            for j in np.arange(i.size()[0]):
                output = Module.forward(i[j][:])
                Module.zero_grad()
                optimizer = optim.SGD(self.module.parameters(), lr=0.01)
                optimizer.zero_grad()
                loss = nn.MSELoss(output,self.testing[j][:])
                loss.backward()
                optimizer.step()
            l=l+1
            trainList.append([output,optimizer,loss])

    def output(self):
        return trainList[:][0]
    def loss(self):
        return trainList[:][2]
    def optimize(self):
        return trainList[:][1]





                


            



