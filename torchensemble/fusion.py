"""
  In fusion-based ensemble methods, the predictions from all base estimators
  are first aggregated as an average output. After then, the training loss is
  computed based on this average output and the ground-truth. The training loss
  is then back-propagated to all base estimators simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseModule


class FusionClassifier(BaseModule):
    """Implementation of the FusionClassifier."""

    def forward(self, X):
        """
        Implementation on the data forwarding process in FusionClassifier.

        Parameters
        ----------
        X : tensor
            Input tensor. Internally, the model will check whether ``X`` is
            compatible with the base estimator.

        Returns
        -------
        proba : tensor
            The predicted probability distribution.
        """
        batch_size = X.size()[0]
        y_pred = torch.zeros(batch_size, self.output_dim).to(self.device)

        # Notice that the output of `FusionClassifier` is different from that
        # of `VotingClassifier` in that the softmax normalization is conducted
        # **after** taking the average of predictions from all base estimators.
        for estimator in self.estimators_:
            y_pred += estimator(X)
        y_pred /= self.n_estimators

        return y_pred

    def fit(self, train_loader):
        """
        Implementation on the training stage of FusionClassifier.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the training data.
        """
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
        """
        Implementation on the evaluating stage of FusionClassifier.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the testing data.
        
        Returns
        -------
        accuracy : float
            The testing accuracy of the fitted model on the ``test_loader``.
        """
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
    """Implementation of the FusionRegressor."""

    def forward(self, X):
        """
        Implementation on the data forwarding process in FusionRegressor.

        Parameters
        ----------
        X : tensor
            Input tensor. Internally, the model will check whether ``X`` is
            compatible with the base estimator.

        Returns
        -------
        pred : tensor
            The predicted values.
        """
        batch_size = X.size()[0]
        pred = torch.zeros(batch_size, self.output_dim).to(self.device)

        for estimator in self.estimators_:
            pred += estimator(X)
        pred /= self.n_estimators

        return pred

    def fit(self, train_loader):
        """
        Implementation on the training stage of FusionRegressor.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the training data.
        """
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
        """
        Implementation on the evaluating stage of FusionClassifier.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the testing data.
        
        Returns
        -------
        mse : float
            The testing mean squared error of the fitted model on the
            ``test_loader``.
        """
        self.eval()
        mse = 0.
        criterion = nn.MSELoss()

        for batch_idx, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            output = self.forward(X_test)

            mse += criterion(output, y_test)

        return mse / len(test_loader)
