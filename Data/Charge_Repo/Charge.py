#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:50:03 2020

@author: Ferran Domingo García

Código complementario al trabajo de fin de grado: IAC, Universidad de la Laguna. 2020. 
Dicho trabajo puede adquirirse contactando con la universidad, para facilitar la interpretación del código.

"""
import torch
import io
import numpy as np
import matplotlib as plt

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

        for i in np.arange(161):
            for j in np.arange(504):
                for k in np.arange(504):
                    MacroT[k,j,i] = ttext[i+j+k]
        MacroT = torch.from_numpy(MacroT)            
        for i in np.arange(161):
            for j in np.arange(504):
                for k in np.arange(504):
                    MacroP[k,j,i] = ftext[i+j+k]
        MacroP = torch.from_numpy(MacroP) 
        for i in np.arange(161):
            MacroZ[i] = ztext[i]
        MacroZ = torch.from_numpy(MacroZ)
        return [MacroT, MacroP, MacroZ]

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

        for i in np.arange(86):
            for j in np.arange(504):
                for k in np.arange(504):
                    MacroTT[k,j,i] = Ttext[i+j+k] 
        MacroTT =  torch.from_numpy(MacroTT)     
        for i in np.arange(86):
            for j in np.arange(504):
                for k in np.arange(504):
                    Macroe[k,j,i] = etext[i+j+k]
        Macroe =  torch.from_numpy(Macroe)
        for i in np.arange(86):
            MacroTA[i] = tatext[i]
        MacroTA =  torch.from_numpy(MacroTA)
        
        return [MacroTT,Macroe, MacroTA]
  
"""
Class used to store in colums the tensors corresponfing to the fiels variables, in order to train neural networks
"""

class LoadSintetic ():
    
    def __init__(self):
    
        self.files = ['./Charge_Repo/File_of_Variables/azimuth.txt', './Charge_Repo/File_of_Variables/field.txt', './Charge_Repo/File_of_Variables/incli.txt', './Charge_Repo/File_of_Variables/pgas.txt', './Charge_Repo/File_of_Variables/rho.txt', './Charge_Repo/File_of_Variables/vel.txt']
        self.columns = self.LoadTable()
            
    def LoadTable (self):
        
        x =list()
        count=0
        Macro = np.zeros((504,504,86))
        
        for i in self.files :
            
            f = open (i)
            data = f.read()
            f.close()
            data = data.split()
    
            for s in np.arange(86):
                for j in np.arange(504):
                    for k in np.arange(504):
                        Macro[k][j][s] = data[k+j+s]
            x.append(torch.from_numpy(Macro))
        return x    
 """
Class that stores the info of all the training models, which are the dictionary states.
"""       

class Model_Reader():
    
    def __init__(self):
        
        with open("/Charge_Repo/Modelpxp.pkl","r") as f:
            self.pxp =pickle.load( f)
        with open("/Charge_Repo/Modelpack.pkl","r") as f:
            self.pack = pickle.load( f)
    
def update_Model (Model_Reader_Object, Loader_Object):
    
    try:
        self.pxp
    try:
        self.pack
                
"""
Class used to train, sintetic models: For that the ntework has to be loades with the correct parametes, once it converge for the training set::

Option 1:
Once trined this class sintetize artificial TAU,E.Pressure and Temperature and attaches them to the cubes
stored in LoadTable / which concatene columns to generate full cubes to compare to numercal values for:  

Option 2:
It uses real data for the training the TAU,Press,Temperature and attach them to the cubes:
    
Option 3: Comprares the results once run Option 1 and Option 2.
""" 
 

"A la classe los objetos se le tienen que pasar inicializados ya - tal que la designación en OB refiere a que el objeto de la calse ya esta creado " 
"Por ahora: testing de la tabla se refiere a muestras reales, que corresponden exatamente a las columnas de LoadSintetic, que se adjuntan a los valores de Pe,Tau,T"
"para formar una tabla con los que ejecutar desire para la simulación nmérica"
"Llamaremos LoadSimulation a los parámetros generados con el programa desire, no confundir con LoadSintetic "

class Sintesis_Training (object):

    def __init__(self, LoadModuleCL_OB,, LoadSintetic_OB , LoadSimulation_OB, Convolutional_TPT):
        
        for i in LoadModuleCL_OB.testing
            LoadSintetic_OB.columns.append(i)
        
        self.filesTable = LoadSintetic_OB.files()
        self.filesTest = LoadSimulation_OB.columns()   
        
        self.Table = LoadSintetic_OB.columns      
        self.Testing = LoadSimulation_OB.columns
        
        self.module = Convolutional_TPT()
        self.netD = {}
        self.losses = list()
    
    def train (self,Convolutional_TPT) :
        
        count =0;
         for i in self.Table:
            self.Table[count] = i.view(-1,86)
            count = count + 1 
            
         for epoch in np.arange(504*504):

            packTest = torch.cat((self.Table[0][epoch][:], self.Table[1][epoch][:],self.Table[2][epoch][:], self.Table[3][epoch][:],self.Table[4][epoch][:],self.Table[5][epoch][:],self.Table[6][epoch][:],self.Table[7][epoch][:],self.Table[8][0][:] ),1)
            packTrain = torch.cat((self.training[0][epoch][:], self.testing[1][epoch][:],self.training[2][0][:]),1)
               
            
        self.netDict.clear()
        self.optimDict.clear()

        for epoch in np.arange(504*504):
 

class  LoadSimulation ():
    
    def __init__(self):
    
        self.files = ['./Desire_Repo/']
        self.columns = self.LoadTable()
            
    def LoadTable (self):
        
        x =list()
        count=0
        Macro = np.zeros((504,504,86))
        
        for i in self.files :
            f = open (i)
            data = f.read()
            f.close()
            data = data.split()
    
            for s in np.arange(86):
                for j in np.arange(504):
                    for k in np.arange(504):
                        Macro[k][j][s] = data[k+j+s]
            x.append(torch.from_numpy(Macro))    
        return x    
            
class Convolutional_TPT (nn.Module):

    def __init__(self):   
        super().__init__():
        self.layer1 =
        self.layer2 =
        
    def forward
        
            
        
        