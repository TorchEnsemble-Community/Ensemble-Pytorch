""" 
  In fusion-based ensemble methods, the predictions from all base estimators are
  first aggregated as an average output. After then, the training loss is 
  computed based on the average output and the ground-truth. The training loss 
  is then back-propagated to all base estimators simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseModule


class FusionClassifier(BaseModule):
    
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)
        
        # Notice that the output of `FusionClassifier` is different from that of
        # `VotingClassifier` in that the softmax normalization is conducted 
        # **after** taking the average of predictions from all base estimators.
        for estimator in self.estimators_:
            y_pred += estimator(X)
        y_pred /= self.n_estimators
        
        return y_pred
    
    def fit(self, train_loader):
        
        self.train()
        self._validate_parameters()
        criterion = nn.CrossEntropyLoss()  # for classification
        
        for epoch in range(self.epochs):
            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                
                batch_size = X_train.size()[0]
                X_train, y_train = (X_train.to(self.device),
                                    y_train.to(self.device))
                
                output = self.forward(X_train)
                loss = criterion(output, y_train)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Print training status
                if batch_idx % self.log_interval == 0:
                    y_pred = F.softmax(output, dim=1).data.max(1)[1]
                    correct = y_pred.eq(y_train.view(-1).data).sum()
                    
                    msg = ('Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f} |'
                           ' Correct: {:d}/{:d}')
                    print(msg.format(epoch, batch_idx, loss, 
                                     correct, batch_size))
    
    def predict(self, test_loader):
        
        self.eval()
        correct = 0.

        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = F.softmax(self.forward(X_test), dim=1)
            y_pred = output.data.max(1)[1]
            
            correct += y_pred.eq(y_test.view(-1).data).sum()
        
        accuracy = 100. * float(correct) / len(test_loader.dataset)
        
        return accuracy


class FusionRegressor(BaseModule):
    
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)
        
        for estimator in self.estimators_:
            y_pred += estimator(X)
        y_pred /= self.n_estimators
        
        return y_pred
    
    def fit(self, train_loader):
        
        self.train()
        self._validate_parameters()
        criterion = nn.MSELoss()  # for regression
        
        for epoch in range(self.epochs):
            for batch_idx, (X_train, y_train) in enumerate(train_loader):
                
                X_train, y_train = (X_train.to(self.device),
                                    y_train.to(self.device))
                
                output = self.forward(X_train)
                loss = criterion(output, y_train)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Print training status
                if batch_idx % self.log_interval == 0:
                    msg = 'Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}'
                    print(msg.format(epoch, batch_idx, loss))
    
    def predict(self, test_loader):
        
        self.eval()
        mse = 0.
        criterion = nn.MSELoss()

        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = self.forward(X_test)
        
            mse += criterion(output, y_test)
        
        return mse / len(test_loader)
