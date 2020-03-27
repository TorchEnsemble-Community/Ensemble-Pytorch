import torch
import torch.nn as nn
import torch.nn.functional as F
from . basemodule import BaseModule


class GradientBoostingClassifier(BaseModule):
    
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)
        for learner in self.learners:
            y_pred += self.args["shrinkage_rate"] * learner(X)
        return y_pred
    
    def _onehot_coding(self, y):
        y = y.view(-1)
        y_onehot = torch.FloatTensor(y.size()[0], self.args["output_dim"]).to(self.device)
        y_onehot.data.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot
    
    """ Compute pseudo residual for classification in Gradient Boosting """
    def _pseudo_residual(self, X, y, learner_idx):
        y_onehot = self._onehot_coding(y)
        output = torch.zeros_like(y_onehot).to(self.device)
        if learner_idx == 0:
            return y_onehot - F.softmax(output, dim=1)
        else:
            for idx in range(learner_idx):
                output += self.args["shrinkage_rate"] * self.learners[idx](X)
            return y_onehot - F.softmax(output, dim=1)
    
    def fit(self, train_loader):
        self.train()
        criterion = nn.MSELoss(reduction="sum")
        
        # In Gradient Boosting, base learners are fitted sequentially
        for learner_idx, learner in enumerate(self.learners):
            
            # Initialize independent optimizer for each base learner to avoid unexpected dependencies
            learner_optimizer = torch.optim.Adam(learner.parameters(),
                                                 lr=self.args["lr"],
                                                 weight_decay=self.args["weight_decay"])
            
            # Fit each base learner in Gradient Boosting
            for epoch in range(self.epochs):
                for batch_idx, (X_train, y_train) in enumerate(train_loader):
                    X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                    y_residual = self._pseudo_residual(X_train, y_train, learner_idx)
                    output = learner(X_train)
                    loss = criterion(output, y_residual)
                    learner_optimizer.zero_grad()
                    loss.backward()
                    learner_optimizer.step()
                
                    # Print training status
                    if batch_idx % self.log_interval == 0:
                        print("Learner: {:d} | Epoch: {:d} | Batch: {:03d} | Learner-RegLoss: {:.5f}".format(
                            learner_idx+1, epoch+1, batch_idx+1, loss))
    
    def evaluate(self, test_loader):
        self.eval()
        correct = 0.
        with torch.no_grad():
            for batch_idx, (X_test, y_test) in enumerate(test_loader):
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                output = self.predict(X_test)
                y_pred = output.data.max(1)[1]
                correct += y_pred.eq(y_test.view(-1).data).sum()
            accuracy = 100. * float(correct) / len(test_loader.dataset)
            print("Testing Accuracy: {:.3f}".format(accuracy))
