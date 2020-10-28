import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    
    def __init__(self, output_dim=1):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(90, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, output_dim)

    def forward(self, X):
        X = X.view(X.size()[0], -1)
        
        output = F.relu(self.linear1(X))
        output = F.dropout(output)
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        
        return output
