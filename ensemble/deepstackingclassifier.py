import torch
import torch.nn as nn
import torch.nn.functional as F
from . basemodule import BaseModule


class DeepStackingClassifier(BaseModule):
    
    def __init__(self, args, learner, learner_args):
        super(BaseModule, self).__init__()
        self.args = args
        self.learner = learner
        self.learners = nn.ModuleList()
        self.epochs = args["epochs"]
        self.output_dim = args["output_dim"]
        self.log_interval = args["log_interval"]
        self.n_estimators = args["n_estimators"]
        self.device = torch.device("cuda" if args["cuda"] else "cpu")
        
        # Initialize base estimators
        learner_args.update({"output_dim": self.output_dim})
        for _ in range(self.n_estimators):
            self.learners.append(learner(learner_args).to(self.device))
    
    def forward(self, X):
        for idx, learner in enumerate(self.learners):
            if idx == 0:
                stacked_output = learner(X)
            elif idx == self.n_estimators - 1:
                learner_input = torch.cat(stacked_output, X, dim=1)
                return learner(learner_input)
            else:
                learner_input = torch.cat(stacked_output, X, dim=1)
                learner_output = learner(learner_input)
                stacked_output = torch.cat(stacked_output, torch.tensor(learner_output))
                