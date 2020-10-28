""" 
  In voting-based ensemble methods, each base estimator is trained independently,
  ad the final prediction takes the average over predictions from all base 
  estimators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . _base import BaseModule


class VotingClassifier(BaseModule):
    
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred_proba = torch.zeros(batch_size, self.output_dim).to(self.device)
        
        # Average over the class distributions predicted from all base estimators
        for estimator in self.estimators_:
            y_pred_proba += F.softmax(estimator(X), dim=1)
        y_pred_proba /= self.n_estimators
        
        return y_pred_proba
    
    def fit(self, train_loader):
        
        self.train()
        self._validate_parameters()
        criterion = nn.CrossEntropyLoss()
        
        # TODO: Parallelization
        for est_idx, estimator in enumerate(self.estimators_):
            
            # Initialize an independent optimizer for each base estimator to 
            # avoid unexpected dependencies.
            estimator_optimizer = torch.optim.Adam(estimator.parameters(),
                                                   lr=self.lr,
                                                   weight_decay=self.weight_decay)
            
            for epoch in range(self.epochs):
                for batch_idx, (X_train, y_train) in enumerate(train_loader):
                
                    batch_size = X_train.size()[0]
                    X_train, y_train = (X_train.to(self.device), 
                                        y_train.to(self.device))
                    
                    output = estimator(X_train)
                    loss = criterion(output, y_train)
            
                    estimator_optimizer.zero_grad()
                    loss.backward()
                    estimator_optimizer.step()
                    
                    # Print training status
                    if batch_idx % self.log_interval == 0:
                        y_pred = F.softmax(output, dim=1).data.max(1)[1]
                        correct = y_pred.eq(y_train.view(-1).data).sum()
                        
                        msg = ('Estimator: {:03d} | Epoch: {:03d} |' 
                               ' Batch: {:03d} | Loss: {:.5f} | Correct:'
                               ' {:d}/{:d}')
                        print(msg.format(est_idx, epoch, batch_idx, loss, 
                                         correct, batch_size))
    
    def predict(self, test_loader):
        
        self.eval()
        correct = 0.

        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = self.forward(X_test)
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(y_test.view(-1).data).sum()
        
        accuracy = 100. * float(correct) / len(test_loader.dataset)

        return accuracy


class VotingRegressor(BaseModule):
    
    def forward(self, X):
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)
        
        # Average over predictions from all base estimators
        for estimator in self.estimators_:
            y_pred += estimator(X)
        y_pred /= self.n_estimators
        
        return y_pred
    
    def fit(self, train_loader):
        
        self.train()
        self._validate_parameters()
        criterion = nn.MSELoss()
        
        for est_idx, estimator in enumerate(self.estimators_):
            
            # Initialize an independent optimizer for each base estimator to 
            # avoid unexpected dependencies.
            estimator_optimizer = torch.optim.Adam(estimator.parameters(),
                                                   lr=self.lr,
                                                   weight_decay=self.weight_decay)
            
            for epoch in range(self.epochs):
                for batch_idx, (X_train, y_train) in enumerate(train_loader):

                    X_train, y_train = (X_train.to(self.device), 
                                        y_train.to(self.device))
                    
                    output = estimator(X_train)
                    loss = criterion(output, y_train)
            
                    estimator_optimizer.zero_grad()
                    loss.backward()
                    estimator_optimizer.step()
                    
                    # Print training status
                    if batch_idx % self.log_interval == 0:
                        msg = ('Estimator: {:03d} | Epoch: {:03d} |'
                               ' Batch: {:03d} | Loss: {:.5f}')
                        print(msg.format(est_idx, epoch, batch_idx, loss))
    
    def predict(self, test_loader):
        
        self.eval()
        mse = 0.
        criterion = nn.MSELoss()

        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = self.forward(X_test)
        
            mse += criterion(output, y_test)
        
        return mse / len(test_loader)
