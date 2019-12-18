import torch
import torch.nn as nn


class Linear(nn.Module):
    
    def __init__(self, args):
        super(Linear, self).__init__()
        self.args = args
        self.linear = nn.Linear(args["input_dim"], self.args["output_dim"])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
    
    def forward(self, input):
        batch_size = input.size()[0]
        input = input.view(batch_size, -1)
        output = self.linear(input)
        return output

    def batch_train(self, input, target):
        self.optimizer.zero_grad()
        output = self.forward(input)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
