import torch
import torch.nn as nn
import torch.nn.functional as F


class Averaging(nn.Module):
    
    def __init__(self, args, learner, learner_args):
        super(Averaging, self).__init__()
        self.args = args
        self.learners = nn.ModuleList()
        self.epochs = args["epochs"]
        self.log_interval = args["log_interval"]
        self.objective  = args["objective"]
        self.output_dim = args["output_dim"]
        self.n_estimators = args["n_estimators"]
        self.device = torch.device("cuda" if args["cuda"] else "cpu")
        self.criterion = nn.MSELoss() if self.objective=="regression" else nn.CrossEntropyLoss()
        
        # Initialize base estimators
        learner_args.update({"output_dim": self.output_dim})
        for _ in range(self.n_estimators):
            self.learners.append(learner(learner_args).to(self.device))
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                          lr=args["lr"], 
                                          weight_decay=args["weight_decay"])
    
    def __repr__(self):
        repr_str = "n_estimators: {:d}\n".format(self.n_estimators)
        repr_str += "objective: {}\n".format(self.objective)
        return repr_str
        
    def __str__(self):
        return self.__repr__()
    
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)
        for learner in self.learners:
            y_pred += learner(X)
        y_pred /= self.n_estimators
        return y_pred
    
    def predict(self, X):
        self.eval()
        return self.forward(X)
    
    def fit(self, train_loader):
        self.train()
        for epoch in range(self.epochs):
            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                batch_size = X_train.size()[0]
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                output = self.forward(X_train)
                loss = self.criterion(output, y_train)
        
                if batch_idx % self.log_interval == 0:
                    y_pred = output.data.max(1)[1]
                    correct = y_pred.eq(y_train.view(-1).data).sum()
                    print("Epoch: {:d} | Batch: {:03d} | Loss: {:.5f} | Correct: {:d}/{:d}".format(
                        epoch+1, batch_idx+1, loss, correct, batch_size))
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def evaluate(self, test_loader):
        self.eval()
        correct = 0.
        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = self.predict(X_test)
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(y_test.view(-1).data).sum()
        accuracy = 100. * float(correct) / len(test_loader.dataset)
        print("Testing Accuracy: {:.3f}".format(accuracy))
            
    # Utility functions
    def _print_loss(self, test_loader):
        self.eval()
        with torch.no_grad():
            for batch_idx, (X_test, y_test) in enumerate(test_loader):
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                output = self.predict(X_test)
                loss = self.criterion(output, y_test)
                print("Batch: {:03d} | Loss: {:.5f}".format(batch_idx, loss))