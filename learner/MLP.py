import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.linear1 = nn.Linear(self.args['input_dim'], 50)
        self.linear2 = nn.Linear(50, 30)
        self.linear3 = nn.Linear(30, self.args['output_dim'])
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])

    def forward(self, input):
        input = input.view(input.size()[0], -1)
        output = F.relu(self.linear1(input))
        output = F.dropout(output)
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output

    def batch_train(self, input, target):
        self.optimizer.zero_grad()
        output = self.forward(input)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
