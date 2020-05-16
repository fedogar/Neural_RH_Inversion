import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
        
        self.C1 = nn.Linear(input_size, hidden_size)
        self.C2 = nn.Linear(hidden_size, hidden_size)
        self.C3 = nn.Linear(hidden_size, hidden_size)
        self.C4 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.relu(self.C1(x))
        out = self.relu(self.C2(out))
        out = self.relu(self.C3(out))
        out = self.C4(out)
            
        return out
    