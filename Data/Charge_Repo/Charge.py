#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:50:03 2020
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
    

class LoadSintetic ():
    
    def __init__(self):
    
        self.files = ['./Charge_Repo/File_of_Variables/azimuth.txt, /Charge_Repo/File_of_Variables/field.txt, /Charge_Repo/File_of_Variables/incli.txt, /Charge_Repo/File_of_Variables/pgas.txt, /Charge_Repo/File_of_Variables/rho.txt, /Charge_Repo/File_of_Variables/vel.txt, /Charge_Repo/File_of_Variables/zeta.txt']
        self.columns = self.LoadTable()
            
    def LoadTable (self):
        
        x = torch.randn(len(self.files),504,504,86)
        count=0
        
        for i in self.files :
            f = open (i)
            f = f.read()
            f = f.split()
    
            for s in np.arange(86):
                for j in np.arange(504):
                    for k in np.arange(504):
                        x[i][k][j][s] = f[k+j+s]
            count =count+1
            
        return x

            
            
            
"""      
class Sintesis_Table (object):

    def __init__(self, LoadModuleCL, LoadSintetic):
    
        self.Table_ascii = self.table_append(LoadModuel.testing, LoadSintetic.columns)
        self.Location = LoadSintetic.files()
        self.Load = 
        self.

    def table_append (self, LoadModuel.testing, LoadSintetic.columns):
        
        for i in LoadModuel.testing:
"""          
            
        
        