#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:50:03 2020

@author: Ferran Domingo García

Código complementario al trabajo de fin de grado: IAC, Universidad de la Laguna. 2020. 
Dicho trabajo puede adquirirse contactando con la universidad, para facilitar la interpretación del código.

"""
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import io
import numpy as np
import matplotlib as plt
import torch.optim as optim

"""
This part is the one for global functions used insede the calsses
"""
#As it's not posible to mount in a remote storage the data we will need to put the 
#Data in rezie manually:

def MakeSimul ():
    
    Table = ChargerTABLE(ChargerOUT,ChargerRest).Table
    Stoke = ChargeMStokes()
    
    Resize = ResizeDesire(Table, Stoke)

class ChargerIN ():

    def __init__(self):
            self.in_ = self.charge()
            self.files_ = ['Look at the SourceCode']

    def charge(self):
    
        f = open ('Charge_Repo/Presion.txt','r')
        t = open ('Charge_Repo/Temperature.txt','r')
        z = open ('Charge_Repo/Zeta.txt','r')

        ftext = f.read()
        ttext = t.read()
        ztext = z.read()

        ftext = ftext.split()
        ttext = ttext.split()
        ztext = ztext.split()

        f.close()
        t.close()
        z.close()

        MacroT = np.zeros((504,504,161))
        MacroP = np.zeros((504,504,161))
        MacroZ = np.zeros(161)
        
        l=-1
        for k in np.arange(161):
            for j in np.arange(504):
                for i in np.arange(504):
                    l=l+1
                    MacroT[i,j,k] = ttext[l]
       
        MacroT = torch.from_numpy(MacroT)
        l=-1            
        for k in np.arange(161):
            for j in np.arange(504):
                for i in np.arange(504):
                    l=l+1
                    MacroP[i,j,k] = ftext[l]
       
        MacroP = torch.from_numpy(MacroP) 
        for i in np.arange(161):
            MacroZ[i] = ztext[i]
        MacroZ = torch.from_numpy(MacroZ)
        return [MacroT.requires_grad_(True), MacroP.requires_grad_(True), MacroZ.requires_grad_(True)]

class ChargerOUT():

    def __init__(self):
        self.files_ = ['Look at the SourceCode']
        self.out_ = self.charge()

    def charge(self):
     
        e = open ('Charge_Repo/ElectronicPresure_tau.txt','r')
        ta = open ('Charge_Repo/tau.txt','r')
        T = open ('Charge_Repo/Temperature_tau.txt','r')

        etext = e.read()
        tatext = ta.read()
        Ttext = T.read()

        etext = etext.split()
        tatext = tatext.split()
        Ttext = Ttext.split()

        e.close()
        ta.close()
        T.close()

        MacroTA = np.zeros(86)
        Macroe= np.zeros((504,504,86))
        MacroTT = np.zeros((504,504,86))

        l=-1
        for i in np.arange(86):
            for j in np.arange(504):
                for k in np.arange(504):
                    l=l+1
                    MacroTT[k,j,i] = Ttext[l] 
        MacroTT =  torch.from_numpy(MacroTT)     
        l=-1
        for i in np.arange(86):
            for j in np.arange(504):
                for k in np.arange(504):
                    l=l+1
                    Macroe[k,j,i] = etext[l]
                    
        Macroe =  torch.from_numpy(Macroe)
        for i in np.arange(86):
            MacroTA[i] = tatext[i]
        MacroTA =  torch.from_numpy(MacroTA)
        
        return [MacroTT.requires_grad_(True),Macroe.requires_grad_(True), MacroTA.requires_grad_(True)]

def paint_Chargers(ChargerIN,ChargerOUT):
    
    
    import matplotlib.pyplot as plt
    
    COUT = ChargerOUT
    CIN = ChargerIN
    
    Table = [CIN.in_[0], CIN.in_[1], COUT.out_[0], COUT.out_[1]]
       
    MacroTT = np.ones((504,504))
    MacroT = np.ones((504,504))
    MacroP = np.ones((504,504))
    MacroPe = np.ones((504,504))
    
    MacroTTM = np.ones((504,504))
    MacroTM = np.ones((504,504))
    MacroPM = np.ones((504,504))
    MacroPeM = np.ones((504,504))
    
    MacroTTm = np.ones((504,504))
    MacroTm = np.ones((504,504))
    MacroPm = np.ones((504,504))
    MacroPem= np.ones((504,504))
    
    MacroTTs = np.ones((504,504))
    MacroTs = np.ones((504,504))
    MacroPs = np.ones((504,504))
    MacroPes = np.ones((504,504))
    
    bads = list()
    
    for i in np.arange(504):
        for j in np.arange(504):
         
            MacroTTM[i][j] = torch.max(Table[2][j][i]).detach().numpy()
            if MacroTTM[i][j] > 10000*MacroTTM[i][j-1]:
                MacroTTM[i][j] = MacroTTM[i][j-1]; bads.append([(i,j)]);   
            MacroTM[i][j] = torch.max(Table[0][j][i]).detach().numpy()
            if MacroTM[i][j] > 10000*MacroTM[i][j-1]:
                MacroTM[i][j] = MacroTM[i][j-1]; bads.append([(i,j)]);    
            MacroPM[i][j] = torch.max(Table[1][j][i]).detach().numpy()
            if MacroPM[i][j] > 10000*MacroPM[i][j-1]:
                MacroPM[i][j] = MacroPM[i][j-1]; bads.append([(i,j)]);     
            MacroPeM[i][j] = torch.max(Table[3][j][i]).detach().numpy()
            if MacroPeM[i][j] > 10000*MacroPeM[i][j-1]:
                MacroPeM[i][j] = MacroPeM[i][j-1]; bads.append([(i,j)]);
                
            
            MacroTT[i][j] = torch.median(Table[2][j][i]).detach().numpy()
            MacroT[i][j] = torch.median(Table[0][j][i]).detach().numpy()
            MacroP[i][j] = torch.median(Table[1][j][i]).detach().numpy()
            MacroPe[i][j] = torch.median(Table[3][j][i]).detach().numpy()
            
            MacroTTs[i][j] = torch.mean(Table[2][j][i]).detach().numpy()
            MacroTs[i][j] = torch.mean(Table[0][j][i]).detach().numpy()
            MacroPs[i][j] = torch.mean(Table[1][j][i]).detach().numpy()
            MacroTs[i][j] = torch.mean(Table[3][j][i]).detach().numpy()
            
            MacroTTm[i][j] = torch.min(Table[2][j][i]).detach().numpy()
            MacroTm[i][j] = torch.min(Table[0][j][i]).detach().numpy()
            MacroPm[i][j] = torch.min(Table[1][j][i]).detach().numpy()
            MacroPem[i][j] = torch.min(Table[3][j][i]).detach().numpy()       
    
    fig, ax  = plt.subplots(2,2)
    im1 = ax[0][0].imshow(MacroP,aspect='equal' )
    ax[0][0].set_title('Median Pressure')
    im2 = ax[1][1].imshow(MacroPM,aspect='equal' )
    ax[1][1].set_title('Max Pressure')
    im3 = ax[1][0].imshow(MacroPm , aspect='equal' )
    ax[1][0].set_title('Min Pressure')
    im4 = ax[0][1].imshow(MacroPs,aspect='equal' )
    ax[0][1].set_title('Mean Pressure')
    plt.colorbar(im4)
    
    fig, ax  = plt.subplots(2,2)
    im1 = ax[0][0].imshow(MacroTT,aspect='equal' )
    ax[0][0].set_title('Median Temperature')
    im2 = ax[1][1].imshow(MacroTTM,aspect='equal' )
    ax[1][1].set_title('Max Temperature')
    im3 = ax[1][0].imshow(MacroTTm,aspect='equal' )
    ax[1][0].set_title('Min Temperature')
    im4 = ax[0][1].imshow(MacroTTs,aspect='equal' )
    ax[0][1].set_title('Mean Temperature')
    plt.colorbar(im4)
    
    fig, ax  = plt.subplots(2,2)
    im1 = ax[0][0].imshow(MacroT,aspect='equal' )
    ax[0][0].set_title('Median Tau Temperature')
    im2 = ax[1][1].imshow(MacroTM,aspect='equal' )
    ax[1][1].set_title('Max Tau Temperature')
    im3 = ax[1][0].imshow(MacroTm,aspect='equal' )
    ax[1][0].set_title('Min Tau Temperature')
    im4 = ax[0][1].imshow(MacroTs,aspect='equal' )
    ax[0][1].set_title('Mean Tau Temperature')
    plt.colorbar(im4)
    
    fig, ax  = plt.subplots(2,2)
    im1 = ax[0][0].imshow(MacroPe,aspect='equal' )
    ax[0][0].set_title('Median Electronic pressure')
    im2 = ax[1][1].imshow(MacroPeM,aspect='equal' )
    ax[1][1].set_title('Max Field Electronic pressure')
    im3 = ax[1][0].imshow(MacroPm,aspect='equal' )
    ax[1][0].set_title('Min Field Electronic pressure')
    im4 = ax[0][1].imshow(MacroPs,aspect='equal' )
    ax[0][1].set_title('Mean Field Electronic pressure')
    plt.colorbar(im4)   
    
    return bads

class LinearNet(nn.Module):
   
    def __init__(self, in_ = 161*3, out_ = 86*3):
        super(LinearNet,self).__init__()
        self.fc1 = nn.Linear(in_,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,100)
        self.fc4 = nn.Linear(100,out_)
    
    def forward(self,x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class LoadModuleLinear (object):
    
    def __init__ (self,ChargerIN,ChargerOUT,LinearNet):
        super(LoadModuleLinear,self).__init__()
        
        #self.lr = 0.3
        #self.MacroSin = []
        
        self.training = ChargerIN().in_
        self.testing = ChargerOUT().out_
        
                #Normalization variables to retrive:::
        self.PnormTest = torch.mean(self.testing[1])
        self.TnormTest = torch.mean(self.testing[0])
        self.PnormTrain = torch.mean(self.training[1])
        self.TnormTrain = torch.mean(self.training[0])
        self.ZnormTrain = torch.mean(self.training[2])
        self.TaunormTest = torch.mean(self.testing[2])
                
        self.filter_();
        self.norm();
              
        self.set_pxp= self.prepare_set_pxp()
        self.train_pxp_lineal()
        self.save1(self.module1,self.optimizer1)
        
        self.set_pack = self.prepare_set_pack(pack = 252)
        self.train_pxp_lineal()
        self.save2(self.module2,self.optimizer2)
        
           
        #Dictionary for the losses
        #self.netDict = {}
        #self.losses = list()
    
    def save1(module,optimizer):
    
        import pickle
    
        with open ('/Charge_Repo/OM-pxp-Linear' , 'w') as f:
            pickle.dump ([module.state_dict(), optimizer.state_dict()] ,f)
        
    def save2(module,optimizer):
    
        import pickle
    
        with open ('/Charge_Repo/OM-pack-Linear' , 'w') as f:
            pickle.dump ([module.state_dict(), optimizer.state_dict()] ,f)
    
    
    def filter_(self):
        
        for j in np.arange(len(self.training[0])):
            for k in np.arange(len(self.training[0][0])):
                
               if torch.max(self.training[j][k]) > 10000*self.TnormTrain:
                   for i in np.arange(3):
                       self.training[i][j][k-1] = self.training[i][j][k-1]
                       self.testing[i][j][k-1] = self.testing[i][j][k-1]
    
    def norm (self):
        
           #We normalize the weights    
        self.testing[2][0] = self.testing[2][0]/self.TaunormTest
        self.training[2][0] = self.training[2][0]/self.ZnormTrain
               
        self.testing[1] = self.testing[1]/self.PnormTest
        self.testing[0] = self.testing[0]/self.TnormTest
                    
        self.training[1] = self.training[1]/self.PnormTrain
        self.training[0]= self.training[0]/self.TnormTrain
        
              #We normalize the weights    
        self.testing[2] = self.testing[2]/self.TaunormTest
        self.training[2] = self.training[2]/self.ZnormTrain
               
        self.testing[1] = self.testing[1]/self.PnormTest
        self.testing[0] = self.testing[0]/self.TnormTest
                    
        self.training[1] = self.training[1]/self.PnormTrain
        self.training[0]= self.training[0]/self.TnormTrain
    
    def prepare_set_pxp (self):

        packTest = list()
        packTrain = list()
        
        for j in [2,0,1]:
            
            if j  == 2 :
                self.testing[2] = (self.testing[2].repeat(504*504,1)).view(504*504,1,86)
                self.training[2] = (self.training[2].repeat(504*504,1)).view(504*504,1,161)
               
                packTest.append(self.testing[2])
                packTrain.append(self.training[2])
            
            else :
                self.training[j] = self.training[j].view(504*504,1,161)
                self.testing[j] = self.testing[j].view(504*504,1,86)
                
                packTest.append(self.testing[j])
                packTrain.append(self.training[j])
                    
        #This form is more easily to manipulate the atches, as you can specify the numer of
        #pixels inside chunk, that gives already a list of the tensors sliced into Batches
        #After that you just need to squeeze the dimension for individual pixels
        #If you do not want to work with individual pixels need to rest (3,batch,86) type tensor
        #So in some sense you can still .view(3,)
        #ut mantaining the 3 chanles for each variable
            
        packTest= torch.cat(packTest[:], dim =1)
        packTest = packTest.chunk(504*504,dim=0)
        
        packTrain= torch.cat(packTrain[:], dim =1)
        packTrain = packTrain.chunk(504*504,dim=0)
        
        
        packTrain = list(packTrain)
        counter =0
        for k in packTrain:
            packTrain[counter]= torch.flatten(k, start_dim = 0)
            counter=counter + 1 
        
        packTest = list(packTest)
        counter =0
        for k in packTest:
            packTest[counter]= torch.flatten(k, start_dim =0)
            counter=counter + 1 
        
        return [packTrain, packTest]
     
    def prepare_set_pack (self,pack):
        
        packTest = list()
        packTrain = list()
        
        for j in [2,0,1]:
            
            if j  == 2 :

                packTest.append(self.testing[2])
                packTrain.append(self.training[2])
            
            else :

                packTest.append(self.testing[j])
                packTrain.append(self.training[j])
                     
        #This form is more easily to manipulate the atches, as you can specify the numer of
        #pixels inside chunk, that gives already a list of the tensors sliced into Batches
        #After that you just need to squeeze the dimension for individual pixels
        #If you do not want to work with individual pixels need to rest (3,batch,86) type tensor
        #So in some sense you can still .view(3,)
        #ut mantaining the 3 chanles for each variable
            
        packTest= torch.cat(packTest[:], dim =1)
        packTest = packTest.chunk(int(504*504/pack),dim=0)
        
        packTrain= torch.cat(packTrain[:], dim =1)
        packTrain = packTrain.chunk(int(504*504/pack),dim=0)
              
        packTrain = list(packTrain)
        packTest = list(packTest)
        
       
       #This just for linear training
            
        for k in np.arange(len(packTest)):
            for j in np.arange(len(packTest[0])):
                packTest[k][j] = packTest[k][j].view(86*3) 
        
        for k in np.arange(len(packTrain)):
            for j in np.arange(len(packTrain[0])):
                packTrain[k][j] = packTrain[k][j].view(161*3)
        
        return [packTrain, packTest]
                
        #Here we chunk the tensors of each variable in order to train 
                
    def train_pxp_lineal(self):
        
        Train = self.set_pxp[0]
        Test = self.set_pxp[1]
        
        self.module1 = LinearNet(in_=len(Train[0]), out_=len(Test[0]))
        self.losses1 = list()

        for epochSet in np.arange(2):
            
            for i in np.arange(len(Test)):
                
                with torch.autograd.set_detect_anomaly(True):   
                    
                    output = self.module.forward(Train[i].clone().detach().requires_grad_(True).float())
                    self.module1.zero_grad()
                    self.optimizer1 = optim.Adam(self.module1.parameters(), 0.3)
                    self.optimizer1.zero_grad()
                    loss =F.mse_loss(output,Test[i].clone().detach().requires_grad_(True).float())
                    
                    if loss == float('Inf'):
                        break 
                    
                    loss.float()
                    self.losses1.append(loss)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    
               
        x = np.asarray(self.losses1)
        plt.loglog(x)
        plt.title('Evolution of the losses by cycle -Module1 Net- pxp: Train')
        plt.xlabel('Epoch')
        plt.ylabel('Losses')
         
        
    def train_pack_lineal(self):
                   
            Train = self.set_pack[0]
            Test = self.set_pack[1]
            
            #For linear training we compact the 3 type variables to a pixel

            self.module2 = LinearNet(in_=len(Train[0][0]), out_=len(Test[0][0]))
            self.losses2 = list()
            for epochSet in np.arange(2):
                
                for i in np.arange(len(Test)):
                    
                    with torch.autograd.set_detect_anomaly(True):
                        
                        output = self.module2.forward(Train[i].clone().detach().requires_grad_(True).float())
                        self.module2.zero_grad()
                        self.optimizer2 = optim.RMSprop(self.module2.parameters(), 0.3)
                        self.optimizer2.zero_grad()
                        loss =F.mse_loss(output,Test[i].clone().detach().requires_grad_(True).float())
                        
                        if loss == float('Inf'):
                            break 
                        
                        loss.float()
                        self.losses2.append(loss)
                        loss.backward()
                        optimizer.step()
                   
            x = np.asarray(self.losses2)
            plt.loglog(x)
            plt.title('Evolution of the losses by cycle -Module2 Net- pack: Train')
            plt.xlabel('Epoch')
            plt.ylabel('Losses')
    
    def sintetize_cube():



class Convolutional1DNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3,4000, stride=2, kernel_size= 5)
        self.conv2 = nn.Conv1d(4000,3, padding = 6, kernel_size = 5)
               
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.conv4(x)
        
        return x
    
class LoadModuleConvol (object):
  
    
    def __init__ (self,ChargerIN,ChargerOUT,Convolutional1DNet):
        super(LoadModuleLConvol,self).__init__()
        
        #self.lr = 0.3
        #self.MacroSin = []
        
        self.training = ChargerIN().in_
        self.testing = ChargerOUT().out_
        
                #Normalization variables to retrive:::
        self.PnormTest = torch.median(self.testing[1])
        self.TnormTest = torch.median(self.testing[0])
        self.PnormTrain = torch.median(self.training[1])
        self.TnormTrain = torch.median(self.training[0])
        self.ZnormTrain = torch.median(self.training[2])
        self.TaunormTest = torch.median(self.testing[2])
           
        self.filter_();
        self.norm();
              
        self.set_pxp= self.prepare_set_pxp()
        self.train_pxp_Conv()
        self.save1(self.module1,self.optimizer1)
        
        self.set_pack = self.prepare_set_pack(pack = 252)
        self.train_pack_Conv()
        self.save2(self.module2,self.optimizer2)
        
   
    def filter_(self):
        
        for j in np.arange(len(self.training[0])):
            for k in np.arange(len(self.training[0][0])): 
                if torch.max(self.training[j][k]) > 10000*self.TnormTrain:
                    for i in np.arange(3):
                        self.training[i][j][k-1] = self.training[i][j][k-1]
                        self.testing[i][j][k-1] = self.testing[i][j][k-1]
    
    def norm (self):
        
        self.testing[2][0] = self.testing[2][0]/self.TaunormTest
        self.training[2][0] = self.training[2][0]/self.ZnormTrain
               
        self.testing[1] = self.testing[1]/self.PnormTest
        self.testing[0] = self.testing[0]/self.TnormTest
                    
        self.training[1] = self.training[1]/self.PnormTrain
        self.training[0]= self.training[0]/self.TnormTrain
        
              #We normalize the weights    
        self.testing[2] = self.testing[2]/self.TaunormTest
        self.training[2] = self.training[2]/self.ZnormTrain
               
        self.testing[1] = self.testing[1]/self.PnormTest
        self.testing[0] = self.testing[0]/self.TnormTest
                    
        self.training[1] = self.training[1]/self.PnormTrain
        self.training[0]= self.training[0]/self.TnormTrain
        
        
    def save1(module,optimizer):
        
        import pickle
        with open ('/Charge_Repo/OM-pxp-Conv' , 'w') as f:
            pickle.dump ([module.state_dict(), optimizer.state_dict()] ,f)

      
    def save2 (module,optimizer):
    
        import pickle
        with open ('/Charge_Repo/OM-pack-Conv' , 'w') as f:
            pickle.dump ([module.state_dict(), optimizer.state_dict()] ,f)
       
        
    def prepare_set_pxp (self):

        packTest = list()
        packTrain = list()
        
        for j in [2,0,1]:
            
            if j  == 2 :
                self.testing[2] = (self.testing[2].repeat(504*504,1)).view(504*504,1,86)
                self.training[2] = (self.training[2].repeat(504*504,1)).view(504*504,1,161)
               
                packTest.append(self.testing[2])
                packTrain.append(self.training[2])
            
            else :
                self.training[j] = self.training[j].view(504*504,1,161)
                self.testing[j] = self.testing[j].view(504*504,1,86)
                
                packTest.append(self.testing[j])
                packTrain.append(self.training[j])
                    
        #This form is more easily to manipulate the atches, as you can specify the numer of
        #pixels inside chunk, that gives already a list of the tensors sliced into Batches
        #After that you just need to squeeze the dimension for individual pixels
        #If you do not want to work with individual pixels need to rest (3,batch,86) type tensor
        #So in some sense you can still .view(3,)
        #ut mantaining the 3 chanles for each variable
            
        packTest= torch.cat(packTest[:], dim =1)
        packTest = packTest.chunk(504*504,dim=0)
        
        packTrain= torch.cat(packTrain[:], dim =1)
        packTrain = packTrain.chunk(504*504,dim=0)
        
        packTrain = list(packTrain)
        packTest = list(packTest)
        
        return [packTrain, packTest]

    def prepare_set_pack (self, pack = 252):

        packTest = list()
        packTrain = list()
        
        for j in [2,0,1]:
            
            if j  == 2 :
                packTest.append(self.testing[2])
                packTrain.append(self.training[2])
            
            else :
    
                packTest.append(self.testing[j])
                packTrain.append(self.training[j])
                    
        #This form is more easily to manipulate the atches, as you can specify the numer of
        #pixels inside chunk, that gives already a list of the tensors sliced into Batches
        #After that you just need to squeeze the dimension for individual pixels
        #If you do not want to work with individual pixels need to rest (3,batch,86) type tensor
        #So in some sense you can still .view(3,)
        #ut mantaining the 3 chanles for each variable
            
        packTest= torch.cat(packTest[:], dim =1)
        packTest = packTest.chunk(int(504*504/pack),dim=0)
        
        packTrain= torch.cat(packTrain[:], dim =1)
        packTrain = packTrain.chunk(int(504*504/pack),dim=0)
        
        packTrain = list(packTrain)
        packTest = list(packTest)
        
        return [packTrain, packTest]

    def train_pxp_conv(self):
        
        Train = self.set_pxp[0]
        Test = self.set_pxp[1]
        
        self.module1 = Convolutional1DNet(in_=len(Train[0]), out_=len(Test[0]))
        self.losses1 = list()
        self.optimizer1 = optim.Adam(self.module1.parameters(), 0.3)
        
        for epochSet in np.arange(2):
            
            for i in np.arange(len(Test)):
                
                with torch.autograd.set_detect_anomaly(True):   
                    
                    output = self.module1.forward(Train[i].clone().detach().requires_grad_(True).float())
                    self.module1.zero_grad()
                    self.optimizer1.zero_grad()
                    loss =F.mse_loss(output,Test[i].clone().detach().requires_grad_(True).float())
                    
                    if loss == float('Inf'):
                        break 
                    
                    loss.float()
                    self.losses1.append(loss)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                                   
        x = np.asarray(self.losses1)
        plt.loglog(x)
        plt.title('Evolution of the losses by cycle -Module1 ConvolNet- pxp: Train')
        plt.xlabel('Epoch')
        plt.ylabel('Losses')
         
    def train_pack_conv(self):
                   
        Train = self.set_pack[0]
        Test = self.set_pack[1]
        
        self.module2 = Convolutional1DNet(in_=len(Train[0][0]), out_=len(Test[0][0]))
        self.losses2 = list()
        self.optimizer2 = optim.Adam(self.module1.parameters(), 0.3)
        
        for epochSet in np.arange(2):
            
            for i in np.arange(len(Test)):
                
                with torch.autograd.set_detect_anomaly(True):   
                    
                    output = self.module1.forward(Train[i].clone().detach().requires_grad_(True).float())
                    self.module2.zero_grad()
                    self.optimizer2.zero_grad()
                    loss =F.mse_loss(output,Test[i].clone().detach().requires_grad_(True).float())
                    
                    if loss == float('Inf'):
                        break 
                    
                    loss.float()
                    self.losses2.append(loss)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                                   
        x = np.asarray(self.losses2)
        plt.loglog(x)
        plt.title('Evolution of the losses by cycle -Module2 ConvolNet- pack: Train')
        plt.xlabel('Epoch')
        plt.ylabel('Losses')

     
class ChargerRest ():
    
    def __init__(self):
    
        self.files = ['./Charge_Repo/File_of_Variables/field.txt', './Charge_Repo/File_of_Variables/vel.txt', './Charge_Repo/File_of_Variables/incli.txt', './Charge_Repo/File_of_Variables/azimuth.txt', './Charge_Repo/File_of_Variables/Zeta.txt', './Charge_Repo/File_of_Variables/pgas.txt', './Charge_Repo/File_of_Variables/rho.txt']
        self.columns = self.LoadTable()
            
    def LoadTable (self):
        
        x =list()
        count=0
        Macro = np.zeros((504,504,86))
          
        import copy
          
        a= np.ones((504,504,86))*0
        a = (torch.from_numpy(a))
        x.append(a.requires_grad_(True))
            
        for i in self.files :
            
            #Linea de cogdigo para generar un micro .txt mediante un valor típico, vamos a tomarlo 1
            f = open (i)
            data = f.read()
            f.close()
            data = data.split()
            
            l=-1
            for k in np.arange(86):
                for j in np.arange(504):
                    for s in np.arange(504):
                        l=l+1
                        Macro[s][j][k] = data[l]
            
            count=count +1
            b = torch.from_numpy(copy.deepcopy(Macro))
            x.append(b.requires_grad_(True))
            
        return x

def representationRest(rest):
           
    import matplotlib.pyplot as plt
    
    names = [ 'micro', 'field', 'vel', 'incli', 'azimut', 'Zeta', 'pgas', 'rho' ]
    fig,ax= plt.subplots(4,sharex=True)
    fig2,ax2= plt.subplots(4,sharex=True)
    for i in np.arange(8):
            if i < 4:
                ax[i].plot(rest.columns[i][300,400].detach())
                ax[i].plot(rest.columns[i][300,400].detach())
                ax[i].set_title(names[i])
            else:
                ax2[i-4].plot(rest.columns[i][300,400].detach())
                ax[i-4].plot(rest.columns[i][300,400].detach())
                ax2[i-4].set_title(names[i])
    
# Desire utiliza en la primera columna appended micro.txt por lo que debemos añadir un archivo:'./Charge_Repo/File_of_Variables/micro.txt',  
#Charge table has the structure of the columns tht desire programe needs for the input data ::::....      

class ChargerTABLE (object):
    
    def __init__(self, ChargerOUT,ChargerRest):
        
        self.ChOUT = ChargerOUT()
        self.ChAppend = ChargerRest()
        self.Table = self.fitData()
        self.paint_Table(self.Table)
        
    
    def paint_Table(Table):    
        import matplotlib.pyplot as plt
        table = np.array(Table); table = np.resize(table,(504,504)) 
        MacroF = np.ones((504,504))
        MacroT = np.ones((504,504))
        MacroP = np.ones((504,504))
        MacroV = np.ones((504,504))    
        MacroFM = np.ones((504,504))
        MacroTM = np.ones((504,504))
        MacroPM = np.ones((504,504))
        MacroVM = np.ones((504,504))     
        MacroFm = np.ones((504,504))
        MacroTm = np.ones((504,504))
        MacroPm = np.ones((504,504))
        MacroVm= np.ones((504,504))      
        MacroFs = np.ones((504,504))
        MacroTs = np.ones((504,504))
        MacroPs = np.ones((504,504))
        MacroVs = np.ones((504,504))     
        for i in np.arange(504):
            for j in np.arange(504):
             
                MacroFM[i][j] = torch.max(table[i][j][4]).detach().numpy()
                MacroTM[i][j] = torch.max(table[i][j][1]).detach().numpy()
                MacroPM[i][j] = torch.max(table[i][j][2]).detach().numpy()
                MacroVM[i][j] = torch.max(table[i][j][5]).detach().numpy()
                
                MacroF[i][j] = torch.median(table[i][j][4]).detach().numpy()
                MacroT[i][j] = torch.median(table[i][j][1]).detach().numpy()
                MacroP[i][j] = torch.median(table[i][j][2]).detach().numpy()
                MacroV[i][j] = torch.median(table[i][j][5]).detach().numpy()
                
                MacroFs[i][j] = torch.mean(table[i][j][4]).detach().numpy()
                MacroTs[i][j] = torch.mean(table[i][j][1]).detach().numpy()
                MacroPs[i][j] = torch.mean(table[i][j][2]).detach().numpy()
                MacroVs[i][j] = torch.mean(table[i][j][5]).detach().numpy()
                
                MacroFm[i][j] = torch.min(table[i][j][4]).detach().numpy()
                MacroTm[i][j] = torch.min(table[i][j][1]).detach().numpy()
                MacroPm[i][j] = torch.min(table[i][j][2]).detach().numpy()
                MacroVm[i][j] = torch.min(table[i][j][5]).detach().numpy()
            
        fig, ax  = plt.subplots(2,2)
        im1 = ax[0][0].imshow(MacroP,aspect='equal' )
        ax[0][0].set_title('Median Pressure')
        im2 = ax[1][1].imshow(MacroPM,aspect='equal' )
        ax[1][1].set_title('Max Pressure')
        im3 = ax[1][0].imshow(MacroPm , aspect='equal' )
        ax[1][0].set_title('Min Pressure')
        im4 = ax[0][1].imshow(MacroPs,aspect='equal' )
        ax[0][1].set_title('Mean Pressure')
        plt.colorbar(im4)        
        fig, ax  = plt.subplots(2,2)
        im1 = ax[0][0].imshow(MacroT,aspect='equal' )
        ax[0][0].set_title('Median Temperature')
        im2 = ax[1][1].imshow(MacroTM,aspect='equal' )
        ax[1][1].set_title('Max Temperature')
        im3 = ax[1][0].imshow(MacroTm,aspect='equal' )
        ax[1][0].set_title('Min Temperature')
        im4 = ax[0][1].imshow(MacroTs,aspect='equal' )
        ax[0][1].set_title('Mean Temperature')
        plt.colorbar(im4)      
        fig, ax  = plt.subplots(2,2)
        im1 = ax[0][0].imshow(MacroV,aspect='equal' )
        ax[0][0].set_title('Median Velocity')
        im2 = ax[1][1].imshow(MacroVM,aspect='equal' )
        ax[1][1].set_title('Max Velocity')
        im3 = ax[1][0].imshow(MacroVm,aspect='equal' )
        ax[1][0].set_title('Min Velocity')
        im4 = ax[0][1].imshow(MacroVs,aspect='equal' )
        ax[0][1].set_title('Mean Velocity')
        plt.colorbar(im4)    
        fig, ax  = plt.subplots(2,2)
        im1 = ax[0][0].imshow(MacroF,aspect='equal' )
        ax[0][0].set_title('Median Field')
        im2 = ax[1][1].imshow(MacroFM,aspect='equal' )
        ax[1][1].set_title('Max Field')
        im3 = ax[1][0].imshow(MacroFm,aspect='equal' )
        ax[1][0].set_title('Min Field')
        im4 = ax[0][1].imshow(MacroFs,aspect='equal' )
        ax[0][1].set_title('Mean Field')
        plt.colorbar(im4)   

        
        
    def fitData (self):
        
        TensorList = list()
        for j in [2,0,1]:
            
            if j  == 2 :
               self.ChOUT.out_[j] = (self.ChOUT.out_[j].repeat(504*504,1)).view(-1,86)
               self.ChOUT.out_[j] = self.ChOUT.out_[j].view(504*504,86)
               TensorList.append(self.ChOUT.out_[j])
            else :
                self.ChOUT.out_[j] = self.ChOUT.out_[j].view(504*504,86)
                TensorList.append(self.ChOUT.out_[j])
            
        for i in np.arange(8):
            self.ChAppend.columns[i] = self.ChAppend.columns[i].view(504*504,86)
            TensorList.append(self.ChAppend.columns[i])
            
        for i in np.arange(len(TensorList)):
            TensorList[i] = TensorList[i].view(1,504*504,86)
            
        TensorList = torch.cat(TensorList[:], dim =0)
        TensorList = TensorList.chunk(504*504,dim=1)
        
        TensorList = list(TensorList)
        
        counter =0
        for k in TensorList:
            TensorList[counter]= torch.squeeze(k, dim = 1)
            counter=counter + 1 
            
        return TensorList   
"""
Notar aquí hemos supesto las profundidades ópticas las mismas para todos los pixeles
Ya que solo nos ha proporcionado eso, al igual que las Z, -- por lo que las P y T de cada modelo de pixel,
Deben de ser obtenidas leyendo dicho valor cuando este llega a dicho valor de pixel.
"""   
#Here manually you can tell the programe how many lines you want to store and to do not have
#to make more comprobartions insert manually the number to check in numberlines and LabelLines

def representationTable(Table):
    
    import matplotlib.pyplot as plt
    
    names = ['logTau', 'Temperature','Presure', 'micro','field,', 'vel', 'incli', 'azimut', 'Zeta', 'pgas', 'rho' ]
    counter =0;
    fig,ax= plt.subplots(4,sharex=True)
    fig2,ax2=plt.subplots(4,sharex=True)
    fig3,ax3=plt.subplots(3,sharex=True)
    
    for i in np.arange(11):
        for j in ([0,2,3,300,4555, 500, 2506]):
            
            if i <= 3 :
                ax[i].plot(Table.Table[j][i].detach())
                ax[i].plot(Table.Table[j][i].detach())
                ax[i].set_title(names[i])
                
            if i>4 and  i<=7:
                ax2[i-4].plot(Table.Table[j][i].detach())
                ax[i-4].plot(Table.Table[j][i].detach())
                ax2[i-4].set_title(names[i])
                
            if i>7: 
                ax3[i-8].plot(Table.Table[j][i].detach())
                ax3[i-8].plot(Table.Table[j][i].detach())
                ax3[i-8].set_title(names[i])
                
class MakeTxt (object):
    
    def __init__ (self, table):
        
        self.Object = table.Table
        self.write()
       
    def line_prepender(self, filename):
        with open(filename, 'r+') as f:
            content = f.read().splitlines(True)
            content[0] = '0.0      1.0     0.0000000E+00\n'
            f.seek(0, 0)
            f.writelines(content)  
            
           # import os
            #bashCommand = "sed -i '' -e '$ d' " + filename
            #os.system(bashCommand)
        
    def write(self) :
           
        #First all the tensors First need to convert to numpy arrays:
        #To be able to writ
        
        from astropy.io import ascii
        import copy
        
        #Then write pixel by pixel
        #for j in np.arange(list(self.Object[0].size())[0]):   
        for j in np.arange(len(self.Object)):  
            filename = './Desire_Repo/ModelCube/pixel' + str(j) + '.mod'
            
            ascii.write(np.transpose(self.Object[j].detach().numpy()), filename, overwrite=True)
            self.line_prepender(filename)
            
#Functionality to run not able, instead go to desire : and in file.py run it with python
#To implement the loop for syntesis of the profiles
def Desire ():
    
    import fileinput
    import os
    names = os.listdir('Desire_Repo/ModelCube/')
    bashCommand = "/Users/ferrannew/anaconda3/envs/pytorch_env/Proyectos_ML/desire/run/example/desire desire.dtrol"
   
    j = names[0].partition(".")[0]
    for i in names: 
        for line in fileinput.input("/Users/ferrannew/anaconda3/envs/pytorch_env/Proyectos_ML/desire/run/example/desire.dtrol"):
            # inside this loop the STDOUT will be redirected to the file
            # the comma after each print statement is needed to avoid double line breaks
            line.replace(j.partition(".")[0], i.partition(".")[0])
            j = i.partition(".")[0]
            os.system(bashCommand)

#Here make the changes to load the file for representation in type of ASCII table, which desire can read
#Change manually in the function pixel in order to store more pixles

class ReadTXT (self):
    return
    
#Here we supose that we will read the data from the ascii tables also;;
#Meaning we can train infinite models if given so ::::
    
class ChargerStoke ():
    
    def __init__(self):
        
        self.lines = self.open_()
        self.number_lines =3;
        self.label_lines = [4,23,24]
        
    def open_(self):
        
        fileO = open('./Desire_Repo/StokeCube/pixel1234.per')
        read = fileO.read()
        fileO.close()
        read = read.split()
            
        keysCa = {'Line': '4' , 'landa' : [],  'I' : [] , 'U' : [], 'V' : [] , 'Q': []}
        keysFe = {'Line': '23','landa' : [], 'I' : [] , 'U' : [], 'V' : [] , 'Q': []}
        keysFeII = {'Line': '24', 'landa' : [],'I' : [] , 'U' : [], 'V' : [] , 'Q': []}
        keys = {'Ca' : keysCa, 'Fe' : keysFe, 'FeII' : keysFeII}
            
        count = 0
        for i in np.arange(len(read)/6,dtype=int):
            if keys['Ca']['Line'] == read[count]:
                keys['Ca']['landa'].append(float(read[count+1]))
                keys['Ca']['I'].append(float(read[count+2]))
                keys['Ca']['U'].append(float(read[count+3]))
                keys['Ca']['V'].append(float(read[count+4]))
                keys['Ca']['Q'].append(float(read[count+5]))
            else:        
                if keys['Fe']['Line'] == read[count]:
                    keys['Fe']['landa'].append(float(read[count+1]))
                    keys['Fe']['I'].append(float(read[count+2]))
                    keys['Fe']['U'].append(float(read[count+3]))
                    keys['Fe']['V'].append(float(read[count+4]))
                    keys['Fe']['Q'].append(float(read[count+5]))
                else:
                    keys['FeII']['Line'] == read[count]
                    keys['FeII']['landa'].append(float(read[count+1]))
                    keys['FeII']['I'].append(float(read[count+2]))
                    keys['FeII']['U'].append(float(read[count+3]))
                    keys['FeII']['V'].append(float(read[count+4]))
                    keys['FeII']['Q'].append(float(read[count+5]))
            count = count+6
        return keys
    
    def paint(self):
        
        names = list(self.lines)
        variables = list(self.lines[names[0]])
        
        import matplotlib.pyplot as plt
        
        for i in names:
            fig, ax = plt.subplots(2,2)

            ax[0,0].set_title(i + '  ' + variables[2])
            ax[0,1].set_title(i + '  ' + variables[3])
            ax[1,0].set_title(i + '  ' + variables[4])
            ax[1,1].set_title(i + '  ' + variables[5])
            ax[0,0].scatter(self.lines[i][variables[1]],self.lines[i][variables[2]])
            ax[0,1].scatter(self.lines[i][variables[1]],self.lines[i][variables[3]])
            ax[1,0].scatter(self.lines[i][variables[1]],self.lines[i][variables[4]])
            ax[1,1].scatter(self.lines[i][variables[1]],self.lines[i][variables[5]])
            
         
            ax[0,0].set_xlabel('landa') 
            ax[0,0].set_ylabel(variables[2])
            ax[0,0].set_ylim(min(self.lines[i][variables[2]]), max(self.lines[i][variables[2]]))
            
            ax[1,1].set_xlabel('landa') 
            ax[1,1].set_ylabel(variables[5])
            ax[1,1].set_ylim(min(self.lines[i][variables[5]]), max(self.lines[i][variables[5]]))
            
          
            ax[1,0].set_xlabel('landa') 
            ax[1,0].set_ylabel(variables[4])
            ax[1,0].set_ylim(min(self.lines[i][variables[4]]), max(self.lines[i][variables[4]]))
            
        
            ax[0,1].set_xlabel('landa') 
            ax[0,1].set_ylabel(variables[3])
            ax[0,1].set_ylim(min(self.lines[i][variables[3]]), max(self.lines[i][variables[3]]))
            
            plt.tight_layout()
  #Class to visualize a .per concrete, but not charge the whole macro    
                       
class ChargeMStokes ():
        #We will use a 504,504 simulations , but can be changed manually:
     
    def __init__(self):
        self.Macro = self.Charge()
        self.paint_Macro()
 
    def Charge (self):
        
        TensorList = list()
        SubtensorList = list()
        
        import os
        import copy
        
        for i in os.listdir('./Desire_Repo/StokeCube/'):
            with open('./Desire_Repo/StokeCube/'+i, 'r') as f:   
                
                read = f.read()
                f.close()
                read = read.split()
                   
                #AA Priori no sabemos como de grande va a ser el mallado asi que la única solución
                #Es emplear una estructura de listas encadenadas
                
                landa = list()
                Q = list()
                V = list()
                U = list()
                I = list()
                
                count = 0
                    
                while '4' == read[count] :
                    landa.append(float(read[count+1]))
                    I.append(float(read[count+2]))
                    U.append(float(read[count+3]))
                    V.append(float(read[count+4]))
                    Q.append(float(read[count+5]))
                    count = count+6
                    
                    if count == int(len(read)):
                        break
                    
                SubtensorList.append(copy.deepcopy([landa,I,U,V,Q]))
                I.clear();Q.clear();U.clear();V.clear();landa.clear();
               
                while '23' == read[count]:
                    landa.append(float(read[count+1]))
                    I.append(float(read[count+2]))
                    U.append(float(read[count+3]))
                    V.append(float(read[count+4]))
                    Q.append(float(read[count+5]))
                    count = count+6
                    
                    if count == int(len(read)):
                        break
                    
                SubtensorList.append(copy.deepcopy([landa,I,U,V,Q]))
                I.clear();Q.clear();U.clear();V.clear();landa.clear();
                    
                while '24' == read[count] :
                    landa.append(float(read[count+1]))
                    I.append(float(read[count+2]))
                    U.append(float(read[count+3]))
                    V.append(float(read[count+4]))
                    Q.append(float(read[count+5]))
                    count = count+6
                    
                    if count == int(len(read)):
                        break
                        
                SubtensorList.append(copy.deepcopy([landa,I,U,V,Q]))
                I.clear();Q.clear();U.clear();V.clear();landa.clear();
                
                for j in np.arange(len(SubtensorList)):
                    SubtensorList[j] = torch.tensor(SubtensorList[j] , dtype = float, requires_grad =True) 
            
            TensorList.append(copy.deepcopy(SubtensorList))
            SubtensorList.clear()
            
        return TensorList
    
    def paint_Macro(self):
    
    import matplotlib.pyplot as plt
    table = np.array(self.Macro); table = np.resize(self.Macro,(504,504))
    MacroF = np.ones((504,504))
    MacroT = np.ones((504,504))
    MacroP = np.ones((504,504))
    MacroV = np.ones((504,504))
    MacroFM = np.ones((504,504))
    MacroTM = np.ones((504,504))
    MacroPM = np.ones((504,504))
    MacroVM = np.ones((504,504))
    MacroFm = np.ones((504,504))
    MacroTm = np.ones((504,504))
    MacroPm = np.ones((504,504))
    MacroVm= np.ones((504,504))
    MacroFs = np.ones((504,504))
    MacroTs = np.ones((504,504))
    MacroPs = np.ones((504,504))
    MacroVs = np.ones((504,504))
    for i in np.arange(504):
        for j in np.arange(504):
         
            MacroFM[i][j] = torch.max(table[i][j][3]).detach().numpy()
            MacroTM[i][j] = torch.max(table[i][j][1]).detach().numpy()
            MacroPM[i][j] = torch.max(table[i][j][2]).detach().numpy()
            MacroVM[i][j] = torch.max(table[i][j][4]).detach().numpy()
            
            MacroF[i][j] = torch.median(table[i][j][3]).detach().numpy()
            MacroT[i][j] = torch.median(table[i][j][1]).detach().numpy()
            MacroP[i][j] = torch.median(table[i][j][2]).detach().numpy()
            MacroV[i][j] = torch.median(table[i][j][4]).detach().numpy()
            
            MacroFs[i][j] = torch.mean(table[i][j][3]).detach().numpy()
            MacroTs[i][j] = torch.mean(table[i][j][1]).detach().numpy()
            MacroPs[i][j] = torch.mean(table[i][j][2]).detach().numpy()
            MacroVs[i][j] = torch.mean(table[i][j][4]).detach().numpy()
            
            MacroFm[i][j] = torch.min(table[i][j][3]).detach().numpy()
            MacroTm[i][j] = torch.min(table[i][j][1]).detach().numpy()
            MacroPm[i][j] = torch.min(table[i][j][2]).detach().numpy()
            MacroVm[i][j] = torch.min(table[i][j][4]).detach().numpy()
            
    fig, ax  = plt.subplots(2,2)
    im1 = ax[0][0].imshow(MacroP,aspect='equal' )
    ax[0][0].set_title('Median V:Stokes')
    im2 = ax[1][1].imshow(MacroPM,aspect='equal' )
    ax[1][1].set_title('Max V:Stokes')
    im3 = ax[1][0].imshow(MacroPm , aspect='equal' )
    ax[1][0].set_title('Min V:Stokes')
    im4 = ax[0][1].imshow(MacroPs,aspect='equal' )
    ax[0][1].set_title('Mean V:Stokes')
    plt.colorbar(im4)
    fig, ax  = plt.subplots(2,2)
    im1 = ax[0][0].imshow(MacroT,aspect='equal' )
    ax[0][0].set_title('Median I:Stokes')
    im2 = ax[1][1].imshow(MacroTM,aspect='equal' )
    ax[1][1].set_title('Max I:Stokes')
    im3 = ax[1][0].imshow(MacroTm,aspect='equal' )
    ax[1][0].set_title('Min I:Stokes')
    im4 = ax[0][1].imshow(MacroTs,aspect='equal' )
    ax[0][1].set_title('Mean I:Stokes')
    plt.colorbar(im4)  
    fig, ax  = plt.subplots(2,2)
    im1 = ax[0][0].imshow(MacroV,aspect='equal' )
    ax[0][0].set_title('Median Q:Stokes')
    im2 = ax[1][1].imshow(MacroVM,aspect='equal' )
    ax[1][1].set_title('Max Q:Stokes')
    im3 = ax[1][0].imshow(MacroVm,aspect='equal' )
    ax[1][0].set_title('Min Q:Stokes')
    im4 = ax[0][1].imshow(MacroVs,aspect='equal' )
    ax[0][1].set_title('Mean Q:Stokes')
    plt.colorbar(im4) 
    fig, ax  = plt.subplots(2,2)
    im1 = ax[0][0].imshow(MacroF,aspect='equal' )
    ax[0][0].set_title('Median U:Stokes')
    im2 = ax[1][1].imshow(MacroFM,aspect='equal' )
    ax[1][1].set_title('Max U:Stokes)
    im3 = ax[1][0].imshow(MacroFm,aspect='equal' )
    ax[1][0].set_title('Min U:Stokes')
    im4 = ax[0][1].imshow(MacroFs,aspect='equal' )
    ax[0][1].set_title('Mean U:Stokes')
    plt.colorbar(im4)   
    
  #Returns a list of pixels with, inside each pixel each line is a tensor which has to be trained
           
#Remeber that has to be passed the table object already create and the object Stoke, 
#Already created, with one to one identification in :
  
#Here we will use a chain of 3 lines so we prepare the set in order to 


"""
Visulalización de los pixeles de la simulación
"""

class ResizeDesire (object):
    
    
    def __init__ (self, Table, Stoke):

        self.training = Table.Table
        self.testing = Stoke.Macro
        self.rtraining = self.normTable()
        self.rtesting  = self.normStokes()
        self.norm()

        
        #Stokes section
        
        #Make many maxi list s linesin a randoom sampling to resize
    
    def normStokes(self):
        
        import copy
        
        listResizeS = list()
        maxilist1= list()
        maxilist2= list()
        maxilist3= list()
        
        i=0
        for j in np.arange(len(self.testing[i][0])):
            
            for i in np.random.random_sample(5000):
                i = i*1000
                i = int(i)
                
                maxilist1.append(self.testing[i][0][j])
                maxilist2.append(self.testing[i][1][j])
                maxilist3.append(self.testing[i][2][j])
            
            max1 = copy.deepcopy(torch.median(torch.abs(torch.stack(maxilist1))).detach().numpy())
            max2 = copy.deepcopy(torch.median(torch.abs(torch.stack(maxilist2))).detach().numpy())
            max3 = copy.deepcopy(torch.median(torch.abs(torch.stack(maxilist3))).detach().numpy())
            
            max1m = copy.deepcopy(torch.mean(torch.abs(torch.stack(maxilist1))).detach().numpy())
            max2m = copy.deepcopy(torch.mean(torch.abs(torch.stack(maxilist2))).detach().numpy())
            max3m = copy.deepcopy(torch.mean(torch.abs(torch.stack(maxilist3))).detach().numpy())
            
            max1ma = copy.deepcopy(torch.max(torch.abs(torch.stack(maxilist1))).detach().numpy())
            max2ma = copy.deepcopy(torch.max(torch.abs(torch.stack(maxilist2))).detach().numpy())
            max3ma = copy.deepcopy(torch.max(torch.abs(torch.stack(maxilist3))).detach().numpy())
            
            maxilist1.clear()
            maxilist2.clear()
            maxilist3.clear()
        
            listResizeS.append([[max1,max1m,max1m],[max2,max2m,max2ma],[max3ma,max2ma,max3ma]])
 
    def paint_S_T( listResizeS = self.rtrining):
        
        import matplotlib.pyplot as plt
        names = ['landa', 'I','Q','U','V' ]
        counter =0;
        fig,ax= plt.subplots(2,sharex=True)
        fig2,ax2=plt.subplots(2,sharex=True)
        fig3,ax3=plt.subplots(1,sharex=True)
        
        for j in np.arange(5):
            for i in np.arange(3):
                if j <= 1 :
                    ax[j].scatter(['median','mean','max'],listResizeS[j][i])
                    ax[j].set_title(names[j])
                    ax[j].set_yscale('log')
                            
                if j>1 and  j<=3:
                    ax2[j-2].scatter(['median','mean','max'],listResizeS[j][i])
                    ax2[j-2].set_title(names[j])
                    ax2[j-2].set_yscale('log')
                            
                if j>3: 
                    ax3.scatter(['median','mean','max'],listResizeS[j][i])
                    ax3.set_title(names[j])
                    ax3.set_yscale('log')
                
        del maxilist1; del maxilist2; del maxilist3   
        return listResizeS
    
    
    def normTable(self):
        #We normalize all the intensities equally in stokes parameters as we assume that
        #The peak for I is bigger than the others
        
        listResizeT = list()
        maxilist = list()
        
        i=0
        for j in np.arange(len(self.training[i])):
            for i in np.random.random_sample(50000):
                i = i*1000
                i = int(i)
                maxilist.append(self.training[i][j])
                
            max1 = copy.deepcopy(torch.median(torch.abs(torch.stack(maxilist))).detach().numpy())
            max1ma = copy.deepcopy(torch.max(torch.abs(torch.stack(maxilist))).detach().numpy())
            max1m = copy.deepcopy(torch.mean(torch.abs(torch.stack(maxilist))).detach().numpy())
            maxlist.clear()
            
            listResizeT.append([max1,max1m,max1ma])
        
    def paint_S_T( listResizeT= self.rtesting):
        
        import matplotlib.pyplot as plt
        names = ['logTau', 'Temperature','Presure', 'micro','field,', 'vel', 'incli', 'azimut', 'Zeta', 'pgas', 'rho' ]
        counter =0;
        fig,ax= plt.subplots(4,sharex=True)
        fig2,ax2=plt.subplots(4,sharex=True)
        fig3,ax3=plt.subplots(3,sharex=True)
        for i in np.arange(11):
            if i <= 3 :
                ax[i].scatter(['median','mean','max'],listResizeT[i][:])
                ax[i].set_title(names[i])
                ax[i].set_yscale('log')
                    
            if i>4 and  i<=7:
                ax2[i-4].scatter(['median','mean','max'],listResizeT[i][:])
                ax2[i-4].set_title(names[i])
                ax2[i-4].set_yscale('log')
                    
            if i>7: 
                ax3[i-8].scatter(['median','mean','max'],listResizeT[i][:])
                ax3[i-8].set_title(names[i])
                ax3[i-8].set_yscale('log')
            
        return listResizeT
    
        
    def norm (self):
        
        for i in np.arange(len(self.testing[0][1])):
            for j in np.arange(len(self.testing)):
                self.testing[j][0][i] = self.testing[j][0][i] / self.rtesting[i][0][1]
                self.testing[j][1][i] = self.testing[j][1][i] / self.rtesting[i][1][1]
                self.testing[j][2][i] = self.testing[j][2][i] / self.rtesting[i][2][1]
        
        for i in np.arange(len(self.training[0])):
            for j in np.arange(len(self.training)):
                self.training[j][i] = self.training[j][i] / self.rtraining[i][1]        
    
    def paint_trialS(self):
    
        self.lines = self.testing[56]
        
        names =['ca','fe','feii']
        variablesa = ['landa', 'I', 'Q', 'V', 'U']
        variables =[0,1,2,3,4,5]
        
        import matplotlib.pyplot as plt
        
        for i in np.arange(len(names)):
            fig, ax = plt.subplots(2,2)
            ax[0,0].set_title(names[i] + '  ' + variablesa[2])
            ax[0,1].set_title(names[i] + '  ' + variablesa[3])
            ax[1,0].set_title(names[i] + '  ' + variablesa[4])
            ax[1,1].set_title(names[i] + '  ' + variablesa[5])
            ax[0,0].scatter(self.lines[i][variables[1]],self.lines[i][variables[2]])
            ax[0,1].scatter(self.lines[i][variables[1]],self.lines[i][variables[3]])
            ax[1,0].scatter(self.lines[i][variables[1]],self.lines[i][variables[4]])
            ax[1,1].scatter(self.lines[i][variables[1]],self.lines[i][variables[5]])
            ax[0,0].set_xlabel('landa') 
            ax[0,0].set_ylabel(variablesa[2])
            ax[0,0].set_ylim(min(self.lines[i][variables[2]]), max(self.lines[i][variables[2]]))
            ax[1,1].set_xlabel('landa') 
            ax[1,1].set_ylabel(variablesa[5])
            ax[1,1].set_ylim(min(self.lines[i][variables[5]]), max(self.lines[i][variables[5]]))
            ax[1,0].set_xlabel('landa') 
            ax[1,0].set_ylabel(variablesa[4])
            ax[1,0].set_ylim(min(self.lines[i][variables[4]]), max(self.lines[i][variables[4]]))
            ax[0,1].set_xlabel('landa') 
            ax[0,1].set_ylabel(variablesa[3])
            ax[0,1].set_ylim(min(self.lines[i][variables[3]]), max(self.lines[i][variables[3]]))
            
            plt.tight_layout()
           
#This requires to have been saved table in Stokes good with pikcle in binary files
#And also have been resized so we first run Resize Desire once picke is saved good and
#After we run LoadTraining with a Resized Object and a LinearNet
class StokesNet(object):        
                 
        def __init__(self, ResizeDesire, LinearNet):  
             
            self.training = ResizeDesire.training
            self.testing = ResizeDesire.testing
            self.normtraining = ResizeDesire.rtraining 
            self.normtesting = ResizeDesire.rtesting 
            
            self.train() 
            self.paint()
    
        def train_pxp_L (self):
    
            self.loss0= list()
            self.loss1 = list()
            self.loss2= list()
            
            for epoch in np.arange(5):
            #It is implict that there are one to pixels in to out for the reining
                counter =0
                for j in self.testing:
                    
                    test0= j[0]; test0= torch.flatten(test0,start_dim=0)
                    test1= j[1]; test1= torch.flatten(test1,start_dim=0)
                    test2= j[2]; test2= torch.flatten(test2,start_dim=0)
                    
                    train = self.training[counter]
                    train= torch.faltten(train, start_dim=0)
                    
                    self.module0 = LinearNet(in_=len(train), out_=len(test0))
                    self.module1 = LinearNet(in_=len(train), out_=len(test1))
                    self.module2 = LinearNet(in_=len(train), out_=len(test2))
                    
                    with torch.autograd.set_detect_anomaly(True):   
                        
                        self.loss0.append(train (train,test0,self.module0))
                        self.loss1.append(train (train,test1,self.module1))
                        self.loss2.append(train (train,test2,self.module2))
                        
                    counter = counter +1

        def train (train,test,module):
            
              train.requires_grad(True)
              test.requires_grad(True)
              train.zero_grad()
              test.zero_grad()
              
              output = module.forward(train.float())
              module.zero_grad()
              optimizer = optim.RMSprop(module.parameters(), 0.3)
              optimizer.zero_grad()
              loss =F.mse_loss(output,test.float())
                        
              loss.float()
              loss.backward()
              optimizer.step()
              
              return loss
       
        def paint(self):
            
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(3,sharex=True)
            ax[0].scatter(np.arange(len(self.loss0)), self.loss0)
            ax[1].scatter(np.arange(len(self.loss0)), self.loss0)
            ax[2].scatter(np.arange(len(self.loss0)), self.loss0)
            
        def save(self):
            
            with open('./model0.pkl',"r") as f:
                self.model0.parameters() =pickle.load(f)
            with open('./model1.pkl',"r") as f:
                self.model1.parameters() =pickle.load(f)
            with open('./model2.pkl',"r") as f:
                self.model2.parameters() =pickle.load(f)
            

class StokesConv(object):  
                                  
            
        #This form is more easily to manipulate the atches, as you can specify the numer of
        #pixels inside chunk, that gives already a list of the tensors sliced into Batches
        #After that you just need to squeeze the dimension for individual pixels
        #If you do not want to work with individual pixels need to rest (3,batch,86) type tensor
        #So in some sense you can still .view(3,)
        #ut mantaining the 3 chanles for each variable
            

""" 
class Model_Reader():
    
    def __init__(self):  
        
        with open("/Charge_Repo/Modelpack.pkl","r") as f:
            self.pack = pickle.load( f)


Class used to train, sintetic models: For that the ntework has to be loades with the correct parametes, once it converge for the training set::

Option 1:
Once trined this class sintetize artificial TAU,E.Pressure and Temperature and attaches them to the cubes
stored in LoadTable / which concatene columns to generate full cubes to compare to numercal values for:  

Option 2:
It uses real data for the training the TAU,Press,Temperature and attach them to the cubes:
    
Option 3: Comprares the results once run Option 1 and Option 2.


def saveTable(Table, Stokes):
    
    import pickle
    
    with open ('/Volumes/Elements/Table_' , 'w') as f:
        pickle.dump (Table.Table,f)
    
    with open ('/Volumes/Elements/Stokes_' , 'w') as f:
        pickle.dump (Stokes.Macro,f)

def readTable():   


""" 
 




"A la classe los objetos se le tienen que pasar inicializados ya - tal que la designación en OB refiere a que el objeto de la calse ya esta creado " 
"Por ahora: testing de la tabla se refiere a muestras reales, que corresponden exatamente a las columnas de LoadSintetic, que se adjuntan a los valores de Pe,Tau,T"
"para formar una tabla con los que ejecutar desire para la simulación nmérica"
"Llamaremos LoadSimulation a los parámetros generados con el programa desire, no confundir con LoadSintetic "
     
        
