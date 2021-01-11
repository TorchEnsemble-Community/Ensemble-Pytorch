"""
  In bagging-based ensemble methods, each base estimator is trained
  independently. In addition, sampling with replacement is conducted on the
  training data batch to encourge the diversity between different base
  estimators in the ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from ._base import BaseModule, torchensemble_model_doc
from . import utils


__author__ = ["Yi-Xuan Xu"]
__all__ = ["BaggingClassifier",
           "BaggingRegressor"]


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
                            is_classification,
                            logger):
    """Private function used to fit base estimators in parallel."""
    optimizer = utils.set_optimizer(estimator, optimizer, lr, weight_decay)

    for batch_idx, (data, target) in enumerate(train_loader):

        batch_size = data.size()[0]
        data, target = data.to(device), target.to(device)

        # Sampling with replacement
        sampling_mask = torch.randint(high=batch_size,
                                      size=(int(batch_size),),
                                      dtype=torch.int64)
        sampling_mask = torch.unique(sampling_mask)  # remove duplicates
        sampling_data = data[sampling_mask]
        sampling_target = target[sampling_mask]

        sampling_output = estimator(sampling_data)
        loss = criterion(sampling_output, sampling_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            if is_classification:
                pred = sampling_output.data.max(1)[1]
                correct = pred.eq(sampling_target.view(-1).data).sum()

                msg = ("Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                       " | Loss: {:.5f} | Correct: {:d}/{:d}")
                logger.info(msg.format(idx, epoch, batch_idx, loss,
                                       correct, sampling_data.size()[0]))
            else:
                msg = ("Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                       " | Loss: {:.5f}")
                logger.info(msg.format(idx, epoch, batch_idx, loss))

    return estimator


@torchensemble_model_doc("""Implementation on the BaggingClassifier.""",
                         "model")
class BaggingClassifier(BaseModule):

    @torchensemble_model_doc(
        """Implementation on the data forwarding in BaggingClassifier.""",
        "classifier_forward")
    def forward(self, X):
        batch_size = X.size()[0]
        proba = torch.zeros(batch_size, self.n_outputs).to(self.device)

        # Take the average over class distributions from all base estimators.
        for estimator in self.estimators_:
            proba += F.softmax(estimator(X), dim=1) / self.n_estimators

        return proba

    @torchensemble_model_doc(
        """Implementation on the training stage of BaggingClassifier.""",
        "fit")
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

        # Instantiate base estimators and set attributes
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, True)
        self._validate_parameters(lr, weight_decay, epochs, log_interval)

        self.train()

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
                        True,
                        self.logger
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
                                utils.save(self, save_dir, self.logger)

                        msg = ("Epoch: {:03d} | Validation Acc: {:.3f}"
                               " % | Historical Best: {:.3f} %")
                        self.logger.info(msg.format(epoch, acc, best_acc))

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(rets)
        if save_model and not test_loader:
            utils.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of BaggingClassifier.""",
        "classifier_predict")
    def predict(self, test_loader):
        self.eval()
        correct = 0.

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()

        accuracy = 100. * float(correct) / len(test_loader.dataset)

        return accuracy


@torchensemble_model_doc("""Implementation on the BaggingRegressor.""",
                         "model")
class BaggingRegressor(BaseModule):

    @torchensemble_model_doc(
        """Implementation on the data forwarding in BaggingRegressor.""",
        "regressor_forward")
    def forward(self, X):
        batch_size = X.size()[0]
        pred = torch.zeros(batch_size, self.n_outputs).to(self.device)

        # Take the average over predictions from all base estimators.
        for estimator in self.estimators_:
            pred += estimator(X) / self.n_estimators

        return pred

    @torchensemble_model_doc(
        """Implementation on the training stage of BaggingRegressor.""",
        "fit")
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

        # Instantiate base estimators and set attributes
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, False)
        self._validate_parameters(lr, weight_decay, epochs, log_interval)

        self.train()

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
                        False,
                        self.logger
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
                                utils.save(self, save_dir, self.logger)

                        msg = ("Epoch: {:03d} | Validation MSE:"
                               " {:.5f} | Historical Best: {:.5f}")
                        self.logger.info(msg.format(epoch, mse, best_mse))

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(rets)
        if save_model and not test_loader:
            utils.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of BaggingRegressor.""",
        "regressor_predict")
    def predict(self, test_loader):
        self.eval()
        mse = 0.
        criterion = nn.MSELoss()

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)

            mse += criterion(output, target)

        return mse / len(test_loader)
