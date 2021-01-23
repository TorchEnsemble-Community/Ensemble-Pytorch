"""
  In bagging-based ensemble, each base estimator is trained independently.
  In addition, sampling with replacement is conducted on the training data
  batch to encourge the diversity between different base estimators in the
  ensemble.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from joblib import Parallel, delayed

from ._base import BaseModule, torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op


__all__ = ["BaggingClassifier",
           "BaggingRegressor"]


def _parallel_fit_per_epoch(train_loader,
                            estimator,
                            optimizer,
                            criterion,
                            idx,
                            epoch,
                            log_interval,
                            device,
                            is_classification):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """

    msg_list = []

    for batch_idx, (data, target) in enumerate(train_loader):

        batch_size = data.size(0)
        data, target = data.to(device), target.to(device)

        # Sampling with replacement
        sampling_mask = torch.randint(high=batch_size,
                                      size=(int(batch_size),),
                                      dtype=torch.int64)
        sampling_mask = torch.unique(sampling_mask)  # remove duplicates
        sampling_data = data[sampling_mask]
        sampling_target = target[sampling_mask]

        optimizer.zero_grad()
        sampling_output = estimator(sampling_data)
        loss = criterion(sampling_output, sampling_target)
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            # Classification
            if is_classification:
                subsample_size = sampling_data.size(0)
                _, predicted = torch.max(sampling_output.data, 1)
                correct = (predicted == sampling_target).sum().item()

                msg = ("Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                       " | Loss: {:.5f} | Correct: {:d}/{:d}")
                msg_list.append(msg.format(idx, epoch, batch_idx, loss,
                                           correct, subsample_size))
            else:
                msg = ("Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                       " | Loss: {:.5f}")
                msg_list.append(msg.format(idx, epoch, batch_idx, loss))

    return estimator, msg_list


@torchensemble_model_doc("""Implementation on the BaggingClassifier.""",
                         "model")
class BaggingClassifier(BaseModule):

    @torchensemble_model_doc(
        """Implementation on the data forwarding in BaggingClassifier.""",
        "classifier_forward")
    def forward(self, x):
        # Take the average over class distributions from all base estimators.
        outputs = [F.softmax(estimator(x), dim=1)
                   for estimator in self.estimators_]
        proba = op.average(outputs)

        return proba

    @torchensemble_model_doc(
        """Set the attributes on optimizer for BaggingClassifier.""",
        "set_optimizer")
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @torchensemble_model_doc(
        """Set the attributes on scheduler for BaggingClassifier.""",
        "set_scheduler")
    def set_scheduler(self, scheduler_name, **kwargs):
        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True

    @torchensemble_model_doc(
        """Implementation on the training stage of BaggingClassifier.""",
        "fit")
    def fit(self,
            train_loader,
            epochs=100,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader, True)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(set_module.set_optimizer(estimators[i],
                                                       self.optimizer_name,
                                                       **self.optimizer_args))

        if self.use_scheduler_:
            schedulers = []
            for i in range(self.n_estimators):
                schedulers.append(set_module.set_scheduler(optimizers[i],
                                                           self.scheduler_name,
                                                           **self.scheduler_args))  # noqa: E501

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.

        # Internal helper function on pesudo forward
        def _forward(estimators, data):
            outputs = [F.softmax(estimator(data), dim=1)
                       for estimator in estimators]
            proba = op.average(outputs)

            return proba

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()
                rets = parallel(delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        estimator,
                        optimizer,
                        criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        True
                    )
                    for idx, (estimator, optimizer) in enumerate(
                            zip(estimators, optimizers))
                )

                estimators = []
                for ret_val in rets:
                    estimators.append(ret_val[0])
                    # Write logging info
                    for msg in ret_val[1]:
                        self.logger.info(msg)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for _, (data, target) in enumerate(test_loader):
                            data = data.to(self.device)
                            target = target.to(self.device)
                            output = _forward(estimators, data)
                            _, predicted = torch.max(output.data, 1)
                            correct += (predicted == target).sum().item()
                            total += target.size(0)
                        acc = 100 * correct / total

                        if acc > best_acc:
                            best_acc = acc
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                io.save(self, save_dir, self.logger)

                        msg = ("Epoch: {:03d} | Validation Acc: {:.3f}"
                               " % | Historical Best: {:.3f} %")
                        self.logger.info(msg.format(epoch, acc, best_acc))

                # Update the scheduler
                if self.use_scheduler_:
                    for i in range(self.n_estimators):
                        schedulers[i].step()

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of BaggingClassifier.""",
        "classifier_predict")
    def predict(self, test_loader):
        self.eval()
        correct = 0
        total = 0

        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        acc = 100 * correct / total

        return acc


@torchensemble_model_doc("""Implementation on the BaggingRegressor.""",
                         "model")
class BaggingRegressor(BaseModule):

    @torchensemble_model_doc(
        """Implementation on the data forwarding in BaggingRegressor.""",
        "regressor_forward")
    def forward(self, x):
        # Take the average over predictions from all base estimators.
        outputs = [estimator(x) for estimator in self.estimators_]
        pred = op.average(outputs)

        return pred

    @torchensemble_model_doc(
        """Set the attributes on optimizer for BaggingRegressor.""",
        "set_optimizer")
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @torchensemble_model_doc(
        """Set the attributes on scheduler for BaggingRegressor.""",
        "set_scheduler")
    def set_scheduler(self, scheduler_name, **kwargs):
        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True

    @torchensemble_model_doc(
        """Implementation on the training stage of BaggingRegressor.""",
        "fit")
    def fit(self,
            train_loader,
            epochs=100,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader, False)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(set_module.set_optimizer(estimators[i],
                                                       self.optimizer_name,
                                                       **self.optimizer_args))

        if self.use_scheduler_:
            schedulers = []
            for i in range(self.n_estimators):
                schedulers.append(set_module.set_scheduler(optimizers[i],
                                                           self.scheduler_name,
                                                           **self.scheduler_args))  # noqa: E501

        # Utils
        criterion = nn.MSELoss()
        best_mse = float("inf")

        # Internal helper function on pesudo forward
        def _forward(estimators, data):
            outputs = [estimator(data) for estimator in estimators]
            pred = op.average(outputs)

            return pred

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()
                rets = parallel(delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        estimator,
                        optimizer,
                        criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        False
                    )
                    for idx, (estimator, optimizer) in enumerate(
                            zip(estimators, optimizers))
                )

                estimators = []
                for ret_val in rets:
                    estimators.append(ret_val[0])
                    # Write logging info
                    for msg in ret_val[1]:
                        self.logger.info(msg)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        mse = 0
                        for _, (data, target) in enumerate(test_loader):
                            data = data.to(self.device)
                            target = target.to(self.device)
                            output = _forward(estimators, data)
                            mse += criterion(output, target)
                        mse /= len(test_loader)

                        if mse < best_mse:
                            best_mse = mse
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                io.save(self, save_dir, self.logger)

                        msg = ("Epoch: {:03d} | Validation MSE:"
                               " {:.5f} | Historical Best: {:.5f}")
                        self.logger.info(msg.format(epoch, mse, best_mse))

                # Update the scheduler
                if self.use_scheduler_:
                    for i in range(self.n_estimators):
                        schedulers[i].step()

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of BaggingRegressor.""",
        "regressor_predict")
    def predict(self, test_loader):
        self.eval()
        mse = 0
        criterion = nn.MSELoss()

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            mse += criterion(output, target)

        return mse / len(test_loader)
