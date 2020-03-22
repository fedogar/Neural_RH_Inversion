"""
Created on Fri Mar  6 14:50:03 2020

@author: ferrannew
"""

import torch.nn as nn
import torch.nn.functional as F

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



