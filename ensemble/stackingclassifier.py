import torch
import torch.nn as nn
import torch.nn.functional as F
from . basemodule import BaseModule


class StackingClassifier(BaseModule):
    
    def __init__(self, args, learner, learner_args):
        super(BaseModule, self).__init__(args, learner, learner_args)
    
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)
        return y_pred
    
    def fit(self, train_loader):
        return
    
    def evaluate(self, test_loader):
        return
