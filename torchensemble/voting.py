"""
  In voting-based ensemble methods, each base estimator is trained
  independently, and the final prediction takes the average over predictions
  from all base estimators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from ._base import BaseModule
from . import utils


__author__ = ["Yi-Xuan Xu"]
__all__ = ["VotingClassifier",
           "VotingRegressor"]


def _parallel_fit_per_epoch(train_loader,
                            lr,
                            weight_decay,
                            epoch,
                            optimizer,
                            log_interval,
                            idx,
                            estimator,
                            criterion,
                            device,
                            verbose,
                            is_classification=True):
    """Private function used to fit base estimators in parallel."""
    optimizer = utils.set_optimizer(estimator, optimizer, lr, weight_decay)

    for batch_idx, (data, target) in enumerate(train_loader):

        batch_size = data.size()[0]
        data, target = data.to(device), target.to(device)

        output = estimator(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0 and verbose > 0:

            # Classification
            if is_classification:
                pred = output.data.max(1)[1]
                correct = pred.eq(target.view(-1).data).sum()

                msg = ("{} Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                       " | Loss: {:.5f} | Correct: {:d}/{:d}")
                print(msg.format(utils.ctime(), idx, epoch, batch_idx, loss,
                                 correct, batch_size))
            # Regression
            else:
                msg = ("{} Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                       " | Loss: {:.5f}")
                print(msg.format(utils.ctime(), idx, epoch, batch_idx, loss))

    return estimator


class VotingClassifier(BaseModule):
    """
    Implementation on the VotingClassifier.

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
        Implementation on the data forwarding in VotingClassifier.

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
        batch_size = X.size()[0]
        proba = torch.zeros(batch_size, self.n_outputs).to(self.device)

        # Take the average over class distributions from all base estimators.
        for estimator in self.estimators_:
            proba += F.softmax(estimator(X), dim=1) / self.n_estimators

        return proba

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
        Implementation on the training stage of VotingClassifier.

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
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, True)

        self.train()
        self._validate_parameters(lr, weight_decay, epochs, log_interval)

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.

        # Internal helper function on pesudo forward
        def _forward(estimators, data):
            batch_size = data.size()[0]
            proba = torch.zeros(batch_size, self.n_outputs).to(self.device)

            for estimator in estimators:
                proba += F.softmax(estimator(data), dim=1) / self.n_estimators

            return proba

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                rets = parallel(delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        lr,
                        weight_decay,
                        epoch,
                        optimizer,
                        log_interval,
                        idx,
                        estimator,
                        criterion,
                        self.device,
                        self.verbose,
                        True
                    )
                    for idx, estimator in enumerate(estimators)
                )

                estimators = rets
                # Validation
                if test_loader:
                    with torch.no_grad():
                        correct = 0.
                        for _, (data, target) in enumerate(test_loader):
                            data, target = (data.to(self.device),
                                            target.to(self.device))
                            output = _forward(estimators, data)
                            pred = output.data.max(1)[1]
                            correct += pred.eq(target.view(-1).data).sum()
                        acc = 100. * float(correct) / len(test_loader.dataset)

                        if acc > best_acc:
                            best_acc = acc
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                utils.save(self, save_dir, self.verbose)

                        if self.verbose > 0:
                            msg = ("{} Epoch: {:03d} | Validation Acc: {:.3f}"
                                   " % | Historical Best: {:.3f} %")
                            print(msg.format(utils.ctime(), epoch, acc,
                                             best_acc))

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(rets)
        if save_model and not test_loader:
            utils.save(self, save_dir, self.verbose)

    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of VotingClassifier.

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
        correct = 0.

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()

        accuracy = 100. * float(correct) / len(test_loader.dataset)

        return accuracy


class VotingRegressor(BaseModule):
    """
    Implementation on the VotingRegressor.

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
        Implementation on the data forwarding in VotingRegressor.

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

        # Take the average over predictions from all base estimators.
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
        Implementation on the training stage of VotingRegressor.

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
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, False)

        self.train()
        self._validate_parameters(lr, weight_decay, epochs, log_interval)

        # Utils
        criterion = nn.MSELoss()
        best_mse = float("inf")

        # Internal helper function on pesudo forward
        def _forward(estimators, data):
            batch_size = data.size()[0]
            pred = torch.zeros(batch_size, self.n_outputs).to(self.device)

            for estimator in estimators:
                pred += estimator(data) / self.n_estimators

            return pred

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                rets = parallel(delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        lr,
                        weight_decay,
                        epoch,
                        optimizer,
                        log_interval,
                        idx,
                        estimator,
                        criterion,
                        self.device,
                        self.verbose,
                        False
                    )
                    for idx, estimator in enumerate(estimators)
                )

                estimators = rets
                # Validation
                if test_loader:
                    with torch.no_grad():
                        mse = 0.
                        for _, (data, target) in enumerate(test_loader):
                            data, target = (data.to(self.device),
                                            target.to(self.device))
                            output = _forward(estimators, data)
                            mse += criterion(output, target)
                        mse /= len(test_loader)

                        if mse < best_mse:
                            best_mse = mse
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                utils.save(self, save_dir, self.verbose)

                        if self.verbose > 0:
                            msg = ("{} Epoch: {:03d} | Validation MSE:"
                                   " {:.5f} | Historical Best: {:.5f}")
                            print(msg.format(utils.ctime(), epoch,
                                             mse, best_mse))

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(rets)
        if save_model and not test_loader:
            utils.save(self, save_dir, self.verbose)

    def predict(self, test_loader):
        """
        Implementation on the evaluating stage of VotingRegressor.

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
