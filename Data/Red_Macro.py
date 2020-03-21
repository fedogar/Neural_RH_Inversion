#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:31:34 2020

@author: ferrannew
"""
"Programa red neuronal para el Macro ::: "

# import pickle
# import numpy as np
# import io

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import torch.nn as nn

class Net (nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        "Uses a 3x3 kernel"
        self.conv1=nn.Conv2d(1,6,3)
        self.conv2 =nn.Conv2d(6,16,3)
        
        "affine transformations"
        self.fc1 =nn.Linear(16* 6*6,120)
        
    def foward(self,x):
        x = F.max.pool2d(F.relu())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        

"Posible acces to the modules atriburtes of the transformation as:: Net.fc1.bias,.weight"

with open('Macro.pkl') as f:  # Python 3: open(..., 'rb')
     MacroZ, MacroTA ,MacroP ,Macroe ,MacroT ,MacroTT  = pickle.load(f)

