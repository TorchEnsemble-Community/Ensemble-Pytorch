"""
  In fusion-based ensemble methods, predictions from all base estimators are
  first aggregated as an average output. After then, the training loss is
  computed based on this average output and the ground-truth. The training loss
  is then back-propagated to all base estimators simultaneously.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseModule
from . import utils


__author__ = ["Yi-Xuan Xu"]
__all__ = ["FusionClassifier",
           "FusionRegressor"]


class FusionClassifier(BaseModule):
    """
    Implementation on the FusionClassifier.

    Parameters
    ----------
    estimator : torch.nn.Module
        The class of base estimator inherited from :mod:`torch.nn.Module`.
    n_estimators : int
        The number of base estimators in the ensemble.
    estimator_args : dict, default=None
        The dictionary of hyper-parameters used to instantiate base
        estimators.
    cuda : bool, default=True

        - If ``True``, use GPU to train and evaluate the ensemble.
        - If ``False``, use CPU to train and evaluate the ensemble.
    n_jobs : int, default=None
        The number of workers for training the ensemble. This
        argument is used for parallel ensemble methods such as
        :mod:`voting` and :mod:`bagging`. Setting it to an integer larger
        than ``1`` enables a total number of ``n_jobs`` base estimators
        to be trained simultaneously.
    verbose : int, default=1
        Control the level on printing logging information.

        - If ``0``, trigger the silent mode
        - If ``1``, basic logging information on the training and
          evaluating status is printed.
        - If ``> 1``, full logging information is printed.

    Attributes
    ----------
    estimators_ : torch.nn.ModuleList
        The internal container that stores all base estimators.

    """

    def _forward(self, X):
        """
        Implementation on the internal data forwarding in FusionClassifier.
        """
        batch_size = X.size()[0]
        proba = torch.zeros(batch_size, self.n_outputs).to(self.device)

        # Average
        for estimator in self.estimators_:
            proba += estimator(X) / self.n_estimators

        return proba

    def forward(self, X):
        """
        Implementation on the data forwarding in FusionClassifier.

        Parameters
        ----------
        X : tensor
            Input batch of data, which should be a valid input data batch for
            base estimators.
    
        Returns
        -------
        proba : tensor of shape (batch_size, n_classes)
            The predicted class distribution.

        """
        proba = self._forward(X)

        return F.softmax(proba, dim=1)

    def fit(self,
            train_loader,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=100,
            optimizer="Adam",
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):
        """
        Implementation on the training stage of FusionClassifier.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the training data.
        lr : float, default=1e-3
            The learning rate of the parameter optimizer.
        weight_decay : float, default=5e-4
            The weight decay of the parameter optimizer.
        epochs : int, default=100
            The number of training epochs.
        optimizer : {"SGD", "Adam", "RMSprop"}, default="Adam"
            The type of parameter optimizer.
        log_interval : int, default=100
            The number of batches to wait before printting the training status.
        test_loader : torch.utils.data.DataLoader, default=None
            A :mod:`DataLoader` container that contains the evaluating data.
    
            - If ``None``, no validation is conducted after each training
              epoch.
            - If not ``None``, the ensemble will be evaluated on this
              dataloader after each training epoch.
        save_model : bool, default=True
            Whether to save the model.
    
            - If test_loader is ``None``, the ensemble trained over ``epochs``
              will be saved.
            - If test_loader is not ``None``, the ensemble with the best
              validation performance will be saved.
        save_dir : string, default=None
            Specify where to save the model.
    
            - If ``None``, the model will be saved in the current directory.
            - If not ``None``, the model will be saved in the specified
              directory: ``save_dir``.

        """

        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, True)
        optimizer = utils.set_optimizer(self, optimizer, lr, weight_decay)

        self.train()
        self._validate_parameters(lr, weight_decay, epochs, log_interval)

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.

        # Training loop
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):

                batch_size = data.size()[0]
                data, target = data.to(self.device), target.to(self.device)

                output = self._forward(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        pred = output.data.max(1)[1]
                        correct = pred.eq(target.view(-1).data).sum()

                        if self.verbose > 0:
                            msg = ("{} Epoch: {:03d} | Batch: {:03d} | Loss:"
                                   " {:.5f} | Correct: {:d}/{:d}")
                            print(msg.format(utils.ctime(), epoch, batch_idx,
                                             loss, correct, batch_size))

            # Validation
            if test_loader:
                with torch.no_grad():
                    correct = 0.
                    for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = (data.to(self.device),
                                        target.to(self.device))
                        output = self.forward(data)
                        pred = output.data.max(1)[1]
                        correct += pred.eq(target.view(-1).data).sum()
                    acc = 100. * float(correct) / len(test_loader.dataset)

                    if acc > best_acc:
                        best_acc = acc
                        if save_model:
                            utils.save(self, save_dir, self.verbose)

                    if self.verbose > 0:
                        msg = ("{} Epoch: {:03d} | Validation Acc: {:.3f}"
                               " % | Historical Best: {:.3f} %")
                        print(msg.format(utils.ctime(), epoch, acc, best_acc))

        if save_model and not test_loader:
            utils.save(self, save_dir, self.verbose)

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
            The testing accuracy of the fitted ensemble on the ``test_loader``.
        """
        self.eval()
        correct = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()

        accuracy = 100. * float(correct) / len(test_loader.dataset)

        return accuracy


class FusionRegressor(BaseModule):
    """
    Implementation on the FusionRegressor.

    Parameters
    ----------
    estimator : torch.nn.Module
        The class of base estimator inherited from :mod:`torch.nn.Module`.
    n_estimators : int
        The number of base estimators in the ensemble.
    estimator_args : dict, default=None
        The dictionary of hyper-parameters used to instantiate base
        estimators.
    cuda : bool, default=True

        - If ``True``, use GPU to train and evaluate the ensemble.
        - If ``False``, use CPU to train and evaluate the ensemble.
    n_jobs : int, default=None
        The number of workers for training the ensemble. This
        argument is used for parallel ensemble methods such as
        :mod:`voting` and :mod:`bagging`. Setting it to an integer larger
        than ``1`` enables a total number of ``n_jobs`` base estimators
        to be trained simultaneously.
    verbose : int, default=1
        Control the level on printing logging information.

        - If ``0``, trigger the silent mode
        - If ``1``, basic logging information on the training and
          evaluating status is printed.
        - If ``> 1``, full logging information is printed.

    Attributes
    ----------
    estimators_ : torch.nn.ModuleList
        The internal container that stores all base estimators.

    """

    def forward(self, X):
        """
        Implementation on the data forwarding in FusionRegressor.

        Parameters
        ----------
        X : tensor
            Input batch of data, which should be a valid input data batch for
            base estimators.

        Returns
        -------
        pred : tensor of shape (batch_size, n_outputs)
            The predicted values.

        """
        batch_size = X.size()[0]
        pred = torch.zeros(batch_size, self.n_outputs).to(self.device)

        for estimator in self.estimators_:
            pred += estimator(X) / self.n_estimators

        return pred

    def fit(self,
            train_loader,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=100,
            optimizer="Adam",
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):
        """
        Implementation on the training stage of FusionRegressor.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            A :mod:`DataLoader` container that contains the training data.
        lr : float, default=1e-3
            The learning rate of the parameter optimizer.
        weight_decay : float, default=5e-4
            The weight decay of the parameter optimizer.
        epochs : int, default=100
            The number of training epochs.
        optimizer : {"SGD", "Adam", "RMSprop"}, default="Adam"
            The type of parameter optimizer.
        log_interval : int, default=100
            The number of batches to wait before printting the training status.
        test_loader : torch.utils.data.DataLoader, default=None
            A :mod:`DataLoader` container that contains the evaluating data.
    
            - If ``None``, no validation is conducted after each training
              epoch.
            - If not ``None``, the ensemble will be evaluated on this
              dataloader after each training epoch.
        save_model : bool, default=True
            Whether to save the model.
    
            - If test_loader is ``None``, the ensemble trained over ``epochs``
              will be saved.
            - If test_loader is not ``None``, the ensemble with the best
              validation performance will be saved.
        save_dir : string, default=None
            Specify where to save the model.
    
            - If ``None``, the model will be saved in the current directory.
            - If not ``None``, the model will be saved in the specified
              directory: ``save_dir``.

        """
        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, False)
        optimizer = utils.set_optimizer(self, optimizer, lr, weight_decay)

        self.train()
        self._validate_parameters(lr, weight_decay, epochs, log_interval)

        # Utils
        criterion = nn.MSELoss()
        best_mse = np.float("inf")

        # Training loop
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(self.device), target.to(self.device)

                output = self.forward(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "{} Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        print(
                            msg.format(utils.ctime(), epoch, batch_idx, loss)
                        )

            # Validation
            if test_loader:
                with torch.no_grad():
                    mse = 0.
                    for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = (data.to(self.device),
                                        target.to(self.device))
                        output = self.forward(data)
                        mse += criterion(output, target)
                    mse /= len(test_loader)

                    if mse < best_mse:
                        best_mse = mse
                        if save_model:
                            utils.save(self, save_dir, self.verbose)

                    if self.verbose > 0:
                        msg = ("{} Epoch: {:03d} | Validation MSE: {:.5f} |"
                               " Historical Best: {:.5f}")
                        print(msg.format(utils.ctime(), epoch, mse, best_mse))

        if save_model and not test_loader:
            utils.save(self, save_dir, self.verbose)

    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of FusionRegressor.

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

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)

            mse += criterion(output, target)

        return mse / len(test_loader)
