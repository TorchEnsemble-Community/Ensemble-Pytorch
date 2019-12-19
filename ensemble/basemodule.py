import torch
import torch.nn as nn


class BaseModule(nn.Module):
    
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
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                          lr=args["lr"], 
                                          weight_decay=args["weight_decay"])
    
    def __repr__(self):
        repr_str = "estimator: {}\nn_estimators: {:d}\noutput_dim: {:d}".format(
            self.learner.__name__, self.n_estimators, self.output_dim)
        return repr_str
        
    def __str__(self):
        return self.__repr__()
    
    def predict(self, X):
        self.eval()
        return self.forward(X)
