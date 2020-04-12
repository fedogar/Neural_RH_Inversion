"""
Created on Fri Mar  6 14:50:03 2020

@author: Ferran Domingo García

Código complementario al trabajo de fin de grado: IAC, Universidad de la Laguna. 2020. 
Dicho trabajo puede adquirirse contactando con la universidad, para facilitar la interpretación del código.

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import argparse
import os
import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import torchvision.utils as vutils
import pickle

from IPython.display import HTML

#LoadMoulde Module
from Charge_Repo.Charge import ChargerIN
from Charge_Repo.Charge import ChargerOUT

"""
El formato de las estructuras de datos con las que se ha trabajado son ficheros de texto, donde la lectura se hace en arrays de 
[504,504,161] para variables en profundidad y [504,504,86] para variables en profundidades ópticas
"""

class LinearNet(nn.Module):
   
    def __init__(self):
        super(LinearNet,self).__init__()
        self.fc1 = nn.Linear(161*3,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,100)
        self.fc4 = nn.Linear(100,86*3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


#Actually is passed linear net, but in main we can pass the model we want with correct match ChargeIN, ChargeOut 
class LoadModuleLr (object):
    
    def __init__ (self,ChargerIN,ChargerOUT,LinearNet):
        super(LoadModuleLr,self).__init__()
        
        self.lr = 0.05
        self.MacroSin = []
        
        self.training = ChargerIN().in_
        self.testing = ChargerOUT().out_
        self.module = LinearNet()
        
        self.netDict = {}
        self.losses = list()
  
    def train (self, lr_):

        self.module.float()
        self.PnormTest = torch.max(self.testing[1])
        self.TnormTest = torch.max(self.testing[0])
        self.PnormTrain = torch.max(self.training[1])
        self.TnormTrain = torch.max(self.training[0])
        
        
        for i in np.arange(3):
            
            self.training[i] = self.training[i].view(-1,161)
            self.testing[i] = self.testing[i].view(-1,86)
               
        self.testing[2][0] = self.testing[2][0]/max(self.testing[2][0])
        self.training[2][0] = self.training[2][0]/max(self.training[2][0])
        
        self.netDict.clear()

        for epochSet in np.arange(20):
            
            for epoch in np.arange(504*504/96):
                epoch = int(epoch) 
                
                if epochSet == 0:
                    
                    self.testing[1][epoch] = self.testing[1][epoch]/self.PnormTest
                    self.testing[0][epoch] = self.testing[0][epoch]/self.TnormTest
                    
                    self.training[1][epoch] = self.training[1][epoch]/self.PnormTrain
                    self.training[0][epoch] = self.training[0][epoch]/self.TnormTrain
                
                    BatchTrain = torch.zeros((96, 161*3))
                    BatchTest = torch.zeros((96, 86*3))            
                
                for pack in np.arange(96) :
                    
                    packTest = torch.cat((self.testing[0][epoch*96 + pack][:], self.testing[1][epoch*96 + pack][:], self.testing[2][0]),0)
                    packTrain = torch.cat((self.training[0][epoch*96 + pack][:], self.training[1][epoch*96 + pack][:], self.training[2][0]),0)
                
                    BatchTest[pack,:]= packTest 
                    BatchTrain[pack,:]= packTrain
                                      
                output = self.module.forward(BatchTrain.float())
                self.module.zero_grad()
                optimizer = optim.RMSprop(self.module.parameters(), lr_)
                optimizer.zero_grad()
                loss =F.mse_loss(output,BatchTest.float())
                    
                if loss == float('Inf'):
                    break 
                    
                loss.float()
                loss.backward()
                optimizer.step()
               
                if epoch == int((504*504/96)-1):
                    
                    try:
                        x = np.asarray(self.losses)
                        plt.loglog(x)
                        plt.title('Evolution of the losses by cycle')
                        plt.xlabel('Epoch')
                        plt.ylabel('Losses')
                        self.losses.append(loss)
                    
                        if j == 19:
                            self.netDict = { 'fc1' : self.module.fc1.weight, 'fc2' : self.module.fc2.weight , 'fc3' : self.modul.fc2.weight , 'fc4' : self.modul.fc2.weight  } 
                    
                    except NameError:
                        print('Error')
          
        #This section of the code assures to back to usual data / by inverse normalization        
        self.testing[2][0] = self.testing[2][0]*max(self.testing[2][0])
        self.training[2][0] = self.training[2][0]*max(self.training[2][0])
        
        for epoch in np.arange(504*504):
            
            self.testing[1][epoch] = self.testing[1][epoch]*self.PnormTest
            self.testing[0][epoch] = self.testing[0][epoch]*self.TnormTest
                    
            self.training[1][epoch] = self.training[1][epoch]*self.PnormTrain
            self.training[0][epoch] = self.training[0][epoch]*self.TnormTrain
                
            
        #Las variables ya estan normalizadas, por lo que para obtener los cubos originales.
        #Multiplicamos por un valor típico, de los máximos:: posibilidad uno. o hacemos
        #Pasar la red no normalizada, ya que el factor multiplicativo para ajustes lineales 
        #Es irrelevante
        
        def sintesis_Macro(self):
            
            Macro = torch.zeros(504*504,3,86)
            
            for epoch in np.arange(504*504):
            
                packTest = torch.cat(self.testing[0][epoch][:], self.testing[1][epoch][:], self.testing[2],0)
                packTrain = torch.cat(self.training[0][epoch][:], self.training[1][epoch][:], self.training[2],0)
            
                output = self.module.forward(packTrain.float())
                
                Macro[epoch][:]=output.view(3,86)
            
            self.MacroSin = Macro
            
        def sintesis_check(self):
            
            packTest = torch.cat((self.testing[0][0][:], self.testing[1][0][:],self.testing[2][0][:]),0)
            packTrain = torch.cat((self.training[0][0][:], self.training[1][0][:],self.training[2][0][:]),0)
            
            output = self.module.forward(packTrain.float())
            
            #Temperature, pressión, Tau es el orden de la salida de los parámetros
            Pixel = [packTest [0:85],packTest[86:172],packTest[173:259]]
            loss =F.mse_loss(output,packTest.float())
            return Pixel,loss
 
        def losse_graph (self):
            
            x = np.asarray(self.losses)
            plt.loglog(x)
            plt.title('Evolution of the losses by cycle')
            plt.xlabel('Epoch')
            plt.ylabel('Losses')
    
        def Save_State (self): 
            with open("./Charge_Repo/ModelLinear.model","w+") as f:
                pickle.dump(self.netDict , f)
       
"""
Section dedicated to the training with 1D convolutional Nets, between Z,T,P to Pe,Tau,T
"""
class Convolutional1DNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(3,2,74)
        self.conv2 = nn.Conv1d(2,3,8,padding=10 )
        self.conv3 = nn.Conv1d(3,2,14)
        self.conv4 = nn.Conv1d(2,3,3)
        
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        
        return x
    
#It needs to be passed this 3 modules in the terminal call one is load
class LoadModuleCL (object):
      
    def __init__(self,ChargerIN,ChargerOUT,Convolutional1DNet):
        self.lr = 0.09
        self.module2 = Convolutional1DNet()
        self.training = ChargerIN().in_
        self.testing = ChargerOUT().out_
        
        self.TnormTrain = []
        self.TnormTest= []
        self.PnormTrain = []
        self.PnormTest = []
        self.Taunorm = []
        self.Znorm = []
        
        self.nDpxp = {}
        self.nDpack = {}
        self.losspack = list()
        self.losspxp = list()
               
        #Now we will use same example of Linear sampling by trying to normalize all the weights through
    
    def testing_PackxPack(self,lr_):
        self.lr = lr_
        
        for i in np.arange(3):
            
            self.training[i] = self.training[i].view(-1,1,161)
            self.testing[i] = self.testing[i].view(-1,1,86)
            
        self.testing[2][0] = self.testing[2][0]/max(self.testing[2][0])
        self.training[2][0] = self.training[2][0]/max(self.training[2][0])
        
        self.Taunorm.append(max(self.testing[2][0]))
        self.Znorm.append(max(self.training[2][0]))
        
            
            
        for epoch in np.arange(3969):
            
            self.testing[1][64*epoch:64*(epoch+1)][:] = self.testing[1][64*epoch:64*(epoch+1)][:]/max(self.testing[1][64*epoch:64*(epoch+1)][:])
            self.testing[0][64*epoch:64*(epoch+1)][:] = self.testing[0][64*epoch:64*(epoch+1)][:]/max(self.testing[0][64*epoch:64*(epoch+1)][:])
            
            self.PnormTest.append(max(self.testing[1][64*epoch:64*(epoch+1)][:]))
            self.TnormTest.append(max(self.testing[0][64*epoch:64*(epoch+1)][:]))
            
            self.training[1][64*epoch:64*(epoch+1)][:] = self.training[1][64*epoch:64*(epoch+1)][:]/max(self.training[1][64*epoch:64*(epoch+1)][:])
            self.training[0][64*epoch:64*(epoch+1)][:]= self.training[0][64*epoch:64*(epoch+1)][:]/max(self.training[0][64*epoch:64*(epoch+1)][:])
             
            self.PnormTest.append(max(self.testing[1][64*epoch:64*(epoch+1)][:]))
            self.TnormTest.append(max(self.testing[0][64*epoch:64*(epoch+1)][:]))
            
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
                self.nDpack = { 'conv1' : self.module2.conv1.weight, 'conv2' : self.module2.conv2.weight, 'conv3' :self.module2.conv3.weight , 'conv4' : self.module2.conv1.weight } 

    def testing_PxP(self,lr_):
        self.lr = lr_
        
        for i in np.arange(3):
            
            self.training[i] = self.training[i].view(-1,1,161)
            self.testing[i] = self.testing[i].view(-1,1,86)
            
        self.testing[2][0] = self.testing[2][0]/max(self.testing[2][0])
        self.training[2][0] = self.training[2][0]/max(self.training[2][0])
        
        self.Taunorm.append(max(self.testing[2][0]))
        self.Znorm.append(max(self.training[2][0]))
        
        for epoch in np.arange(504*504):
            #In this form take convolution pixel to pixel / does not allow correlations between pixels ::
            
            if epoch == 0:
                
                self.PnormTest.append(torch.max(self.testing[1]))
                self.TnormTest.append(torch.max(self.testing[0]))
                
                self.testing[1][epoch] = self.testing[1][epoch]/torch.max(self.testing[1])
                self.testing[0][epoch] = self.testing[0][epoch]/torch.max(self.testing[0])
                
                self.PnormTrain.append(torch.max(self.training[1]))
                self.TnormTrain.append(torch.max(self.training[0]))
            
                self.training[1][epoch] = self.training[1][epoch]/torch.max(self.training[1][epoch])
                self.training[0][epoch] = self.training[0][epoch]/torch.max(self.training[0][epoch])
            
            else:
                 
                 self.training[1][epoch] = self.training[1][epoch]/self.PnormTrain
                 self.training[0][epoch] = self.training[0][epoch]/self.TnormTrain
            
                 self.testing[1][epoch] = self.testing[1][epoch]/self.PnormTest
                 self.testing[0][epoch] = self.testing[0][epoch]/self.TnormTest
                
            
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
                self.nDpxp = { 'conv1' : self.module2.conv1.weight, 'conv2' : self.module2.conv2.weight, 'conv3' :self.module2.conv3.weight , 'conv4' : self.module2.conv1.weight } 
    
    
    def sinstesis_check(self):
            
            packTest = torch.cat((self.testing[0][0][:], self.testing[1][0][:],self.testing[2][0][:]),0)
            packTrain = torch.cat((self.training[0][0][:], self.training[1][0][:],self.training[2][0][:]),0)
            
            output = self.module2.forward(packTrain.float())
            
            #Temperature, pressión, Tau es el orden de la salida de los parámetros
            Pixel = [packTest [0:85],packTest[86:172],packTest[173:259]]
            loss =F.mse_loss(output,packTest.float())
            return Pixel,loss
    
            
    def Save_State (self):
        with open("/Charge_Repo/Modelpxp.model","w+") as f:
             pickle.dump(self.nDpxp , f)
        with open("/Charge_Repo/Modelpack.model","a+") as f:
              pickle.dump(self.nDpack, f)
    
"""
Section dedicated to conversion for T,P,Tau +list to sintetic models crafted by desire
"""


class ConvlutionalTPT (nn.Molude):

        def __init__(self,):
            self.fc1 = nn.Linear(, out_features)
            self.fc2 = nn.Linear(in_features, out_features)
            self.fc3 = nn.Linear(in_features, out_features)
            self.fc4 = nn.Linear(in_features, out_features)
        
"""
"La red se puede entrenar con modelos sinteticos o reales para ver las diferencias:::"


class ConvolutionalTPT (n.Module):
    
    def __init__ ():
        super().__init__():
        
        self.fc1 = nn.Convolutiovlutional1DNet()
        self.fc2 = nn.Convolutiovlutional1DNet()
        self.fc3 = nn.Convolutiovlutional1DNet()
        self.fc4 =

#Introcucimos un objeto modelo entrenado tipo LinearNetLr
class LoadModuleTPT(object):
    
     def __init__(self, LinearNetLr, ConvolutionalTPT):
        self.module = ConvolutionalTPT()
        self.sintetic_cube = LinearNetLr.Sintesis
        
        self.nDpxp = {}
        self.nDpack = {}
        
        def testingReal (self):
            
        """ 
        def testingSin (self):
        """

    def Load_State (self):
        
        with open("/Charge_Repo/Modelpxp.model","r+") as f:
             LoadModuleCL.nDpxp  = pickle.load(f)
        with open("/Charge_Repo/Modelpack.model","r+") as f:
             LoadModuleCL.nDpack = pickle.load(f)


"""                                      
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
"""
Here we define the classes related to the overfitting, underfitting checking for the different examples

from sklearn.model_selection import ShuffleSplit

def ValidationTest (object):
    
    ss = ShuffleSplit(n_splits=2540156, test_size=0.25, random_state=0)
    cl = LoadModuleCL(ChargerIN,ChargerOUT,Convolutional1DNet,LinearNet)
    for i in no.arange(3):
        cl.testing[i].view(-1,1,86)     
        cl.training[i](-1,1,161)
    
 """   
    
    



            



