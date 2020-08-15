import torch
import torch.nn as nn
import torch.nn.functional as F
from . basemodule import BaseModule


class BaggingClassifier(BaseModule):
    
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)
        for learner in self.learners:
            y_pred += learner(X)
        y_pred /= self.n_estimators
        return y_pred
    
    def fit(self, train_loader):
        self.train()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                batch_size = X_train.size()[0]
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                loss = torch.tensor(0.).to(self.device)
                
                # In bagging, each base learner is fitted on one sampled batch of data
                for learner in self.learners:
                    sampled_mask = torch.randint(high=batch_size, size=(int(batch_size),), dtype=torch.int64)
                    sampled_X_train = X_train[sampled_mask]
                    sampled_y_train = y_train[sampled_mask]
                    sampled_output = learner(sampled_X_train)
                    loss += criterion(sampled_output, sampled_y_train)
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Print training status
                if batch_idx % self.log_interval == 0:
                    with torch.no_grad():
                        output = F.softmax(self.forward(X_train), dim=1)
                        y_pred = output.data.max(1)[1]
                        correct = y_pred.eq(y_train.view(-1).data).sum()
                        print("Epoch: {:d} | Batch: {:03d} | Loss: {:.5f} | Correct: {:d}/{:d}".format(
                            epoch+1, batch_idx+1, loss, correct, batch_size))
    
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
