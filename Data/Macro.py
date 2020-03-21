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

f = open ('Presion.txt','r')
t = open ('Temperature.txt','r')
z = open ('Zeta.txt','r')

ftext = f.read()
ttext = t.read()
ztext = z.read()

ftext = ftext.split()
ttext = ttext.split()
ztext = ztext.split()

f.close();
t.close();
z.close();

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
            
e = open ('ElectronicPresure_tau.txt','r')
ta = open ('tau.txt','r')
T = open ('Temperature_tau.txt','r')

etext = e.read()
tatext = ta.read()
Ttext = T.read()

etext = etext.split()
tatext = tatext.split()
Ttext = Ttext.split()

e.close();
ta.close();
T.close();

MacroTA = np.zeros(86);
Macroe= np.zeros((504,504,86));
MacroTT = np.zeros((504,504,86));

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

           
import pickle
# obj0, obj1, obj2 are created here..
# Saving the objects:
with open('Macro.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([ MacroZ, MacroTA ,MacroP ,Macroe ,MacroT ,MacroTT ], f)

# Getting back the objects:
with open('Macro.pkl') as f:  # Python 3: open(..., 'rb')
     MacroZ, MacroTA ,MacroP ,Macroe ,MacroT ,MacroTT  = pickle.load(f)