import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F 

class Net (nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(4,4)
        self.fc2 = nn.Linear(4,16)
        self.fc3 = nn.Linear(16,3)           

    def forward (self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

net = Net()
print (net)

params = list(net.parameters())
print(len(params))

a = torch.rand(1,1,22)
output =net(a)







"Optimizer"
import torch.optim as optim
optimizer = optim.SGD(net.parameters(),lr =0.01)
optimizer.zero_grad()
optimizer.



