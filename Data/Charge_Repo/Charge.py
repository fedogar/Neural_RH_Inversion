#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:50:03 2020

@author: ferrannew
"""
import torch, torchvision
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
                      
        for i in np.arange(161):
            for j in np.arange(504):
                for k in np.arange(504):
                    MacroP[k,j,i] = ftext[i+j+k]
            
        for i in np.arange(161):
            MacroZ[i] = ztext[i]

        return [MactoT, MacroP, MacroZ]

class ChargerOUT():

    def __init__(self):
        self.files_ = ['Look at the SourceCode']
        self.in_ = self.charge()

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
        for i in np.arange(86):
            for j in np.arange(504):
                for k in np.arange(504):
                    Macroe[k,j,i] = etext[i+j+k]    
        for i in np.arange(86):
            MacroTA[i] = tatext[i]
        return [MacroTT,Macroe, MacroTA]
        