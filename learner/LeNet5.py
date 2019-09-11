import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.args["output_dim"])
        
    #    self.criterion = nn.MSELoss()
    #    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
    
    def forward(self, data):
        output = F.max_pool2d(F.relu(self.conv1(data)), (2, 2))
        output = F.max_pool2d(F.relu(self.conv2(output)), (2, 2))
        output = output.view(-1, self.num_flat_features(output))
        output = F.relu(self.fc1(output))
        output = F.dropout(output)
        output = F.relu(self.fc2(output))
        output = F.dropout(output)
        output = self.fc3(output)
        return output
   
    def num_flat_features(self, data):
        size = data.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
   
    def batch_train(self, input, target):
        self.optimizer.zero_grad()
        output = self.forward(input)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
