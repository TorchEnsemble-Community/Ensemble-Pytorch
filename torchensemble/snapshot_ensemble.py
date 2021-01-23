"""
  Snapshot ensemble generates many base estimators by enforcing a base
  estimator to converge to its local minima many times and save the
  model parameters at that point as a snapshot. The final prediction takes
  the average over predictions from all snapshot models.

  Reference:
      G. Huang, Y.-X. Li, G. Pleiss et al., Snapshot Ensemble: Train 1, and
      M for free, ICLR, 2017.
"""


import copy
import math
import torch
import logging
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from ._base import BaseModule, torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op


__all__ = ["_BaseSnapshotEnsemble",
           "SnapshotEnsembleClassifier",
           "SnapshotEnsembleRegressor"]


__set_optimizer_doc = """
    Parameters
    ----------
    optimizer_name : string
        The name of the optimizer, should be one of {"Adadelta","Adagrad",
        "Adam","AdamW","SparseAdam","Adamax","ASGD","RMSprop","Rprop","SGD"}.
    **kwargs : keyword arguments
        Miscellaneous keyword arguments on setting the optimizer, should be in
        the form: "lr=1e-3, weight_decay=5e-4, ...". These keyword arguments
        will be directly passed to the :mod:`torch.optim.Optimizer`.
"""


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A :mod:`DataLoader` container that contains the training data.
    lr_clip : list or tuple, default=None
        Specify the accepted range of learning rate. When the learning rate
        determined by the scheduler is out of this range, it will be clipped.

        - The first element should be the lower bound of learning rate.
        - The second element should be the upper bound of learning rate.
    epochs : int, default=100
        The number of training epochs.
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

        - If test_loader is ``None``, the ensemble containing
          ``n_estimators`` base estimators will be saved.
        - If test_loader is not ``None``, the ensemble with the best
          validation performance will be saved.
    save_dir : string, default=None
        Specify where to save the model.

        - If ``None``, the model will be saved in the current directory.
        - If not ``None``, the model will be saved in the specified
          directory: ``save_dir``.
"""


def _snapshot_ensemble_model_doc(header, item="fit"):
    """
    Decorator on obtaining documentation for different snapshot ensemble
    models.
    """
    def get_doc(item):
        """Return selected item"""
        __doc = {"fit": __fit_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + "\n\n"]
        doc.extend(get_doc(item))
        cls.__doc__ = "".join(doc)
        return cls
    return adddoc


class _BaseSnapshotEnsemble(BaseModule):

    def __init__(self,
                 estimator,
                 n_estimators,
                 estimator_args=None,
                 cuda=True):
        super(BaseModule, self).__init__()

        # Make sure estimator is not an instance
        if not isinstance(estimator, type):
            msg = ("The input argument `estimator` should be a class"
                   " inherited from nn.Module. Perhaps you have passed"
                   " an instance of that class into the ensemble.")
            raise RuntimeError(msg)

        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args
        self.device = torch.device("cuda" if cuda else "cpu")
        self.logger = logging.getLogger()

        self.estimators_ = nn.ModuleList()

    def _validate_parameters(self, lr_clip, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""

        if lr_clip:
            if not (isinstance(lr_clip, list) or isinstance(lr_clip, tuple)):
                msg = "lr_clip should be a list or tuple with two elements."
                self.logger.error(msg)
                raise ValueError(msg)

            if len(lr_clip) != 2:
                msg = ("lr_clip should only have two elements, one for lower"
                       " bound, and another for upper bound.")
                self.logger.error(msg)
                raise ValueError(msg)

            if not lr_clip[0] < lr_clip[1]:
                msg = ("The first element = {} should be smaller than the"
                       " second element = {} in lr_clip.")
                self.logger.error(msg.format(lr_clip[0], lr_clip[1]))
                raise ValueError(msg.format(lr_clip[0], lr_clip[1]))

        if not epochs > 0:
            msg = ("The number of training epochs = {} should be strictly"
                   " positive.")
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))

        if not log_interval > 0:
            msg = ("The number of batches to wait before printting the"
                   " training status should be strictly positive, but got {}"
                   " instead.")
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))

        if not epochs % self.n_estimators == 0:
            msg = ("The number of training epochs = {} should be a multiple"
                   " of n_estimators = {}.")
            self.logger.error(msg.format(epochs, self.n_estimators))
            raise ValueError(msg.format(epochs, self.n_estimators))

    def _forward(self, x):
        """
        Implementation on the internal data forwarding in snapshot ensemble.
        """
        # Average
        results = [estimator(x) for estimator in self.estimators_]
        output = op.average(results)

        return output

    def _clip_lr(self, optimizer, lr_clip):
        """Clip the learning rate of the optimizer according to `lr_clip`."""
        if not lr_clip:
            return optimizer

        for param_group in optimizer.param_groups:
            if param_group["lr"] < lr_clip[0]:
                param_group["lr"] = lr_clip[0]
            if param_group["lr"] > lr_clip[1]:
                param_group["lr"] = lr_clip[1]

        return optimizer

    def _set_scheduler(self, optimizer, n_iters):
        """
        Set the learning rate scheduler for snapshot ensemble.
        Please refer to the equation (2) in original paper for details.
        """
        T_M = math.ceil(n_iters / self.n_estimators)
        lr_lambda = lambda iteration: 0.5 * (  # noqa: E731
            torch.cos(torch.tensor(math.pi * (iteration % T_M) / T_M)) + 1
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return scheduler

    def set_scheduler(self, scheduler_name, **kwargs):
        msg = ("The learning rate scheduler for Snapshot Ensemble will be"
               " automatically set. Calling this function has no effect on"
               " the training stage of Snapshot Ensemble.")
        warnings.warn(msg, RuntimeWarning)


@torchensemble_model_doc(
    """Implementation on the SnapshotEnsembleClassifier.""", "model")
class SnapshotEnsembleClassifier(_BaseSnapshotEnsemble):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_classification = True

    @torchensemble_model_doc(
        """Implementation on the data forwarding in SnapshotEnsembleClassifier.""",  # noqa: E501
        "classifier_forward")
    def forward(self, x):
        proba = self._forward(x)

        return F.softmax(proba, dim=1)

    @torchensemble_model_doc(
        """Set the attributes on optimizer for SnapshotEnsembleClassifier.""",
        "set_optimizer")
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @_snapshot_ensemble_model_doc(
        """Implementation on the training stage of SnapshotEnsembleClassifier.""",  # noqa: E501
        "fit"
    )
    def fit(self,
            train_loader,
            lr_clip=None,
            epochs=100,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):
        self._validate_parameters(lr_clip, epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader,
                                                self.is_classification)

        # A dummy model used to generate snapshot ensembles
        estimator_ = self._make_estimator()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(estimator_,
                                             self.optimizer_name,
                                             **self.optimizer_args)

        scheduler = self._set_scheduler(optimizer, epochs * len(train_loader))

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.
        counter = 0  # a counter on generating snapshots
        n_iters_per_estimator = epochs * len(train_loader) // self.n_estimators

        # Training loop
        estimator_.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):

                batch_size = data.size(0)
                data, target = data.to(self.device), target.to(self.device)

                # Clip the learning rate
                optimizer = self._clip_lr(optimizer, lr_clip)

                optimizer.zero_grad()
                output = estimator_(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        _, predicted = torch.max(output.data, 1)
                        correct = (predicted == target).sum().item()

                        msg = ("lr: {:.5f} | Epoch: {:03d} | Batch: {:03d} |"
                               " Loss: {:.5f} | Correct: {:d}/{:d}")
                        self.logger.info(
                            msg.format(
                                optimizer.param_groups[0]["lr"],
                                epoch,
                                batch_idx,
                                loss,
                                correct,
                                batch_size
                            )
                        )

                # Snapshot ensemble updates the learning rate per iteration
                # instead of per epoch.
                scheduler.step()
                counter += 1

            if counter % n_iters_per_estimator == 0:

                # Generate and save the snapshot
                snapshot = copy.deepcopy(estimator_)
                self.estimators_.append(snapshot)

                msg = "Save the snapshot model with index: {}"
                self.logger.info(msg.format(len(self.estimators_) - 1))

            # Validation after each snapshot model being generated
            if test_loader and counter % n_iters_per_estimator == 0:
                self.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for _, (data, target) in enumerate(test_loader):
                        data = data.to(self.device)
                        target = target.to(self.device)
                        output = self.forward(data)
                        _, predicted = torch.max(output.data, 1)
                        correct += (predicted == target).sum().item()
                        total += target.size(0)
                    acc = 100 * correct / total

                    if acc > best_acc:
                        best_acc = acc
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = ("n_estimators: {} | Validation Acc: {:.3f} %"
                           " | Historical Best: {:.3f} %")
                    self.logger.info(
                        msg.format(
                            len(self.estimators_),
                            acc,
                            best_acc
                        )
                    )

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of SnapshotEnsembleClassifier.""",  # noqa: E501
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


@torchensemble_model_doc(
    """Implementation on the SnapshotEnsembleRegressor.""", "model")
class SnapshotEnsembleRegressor(_BaseSnapshotEnsemble):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_classification = False

    @torchensemble_model_doc(
        """Implementation on the data forwarding in SnapshotEnsembleRegressor.""",  # noqa: E501
        "regressor_forward")
    def forward(self, x):
        pred = self._forward(x)
        return pred

    @torchensemble_model_doc(
        """Set the attributes on optimizer for SnapshotEnsembleRegressor.""",
        "set_optimizer")
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @_snapshot_ensemble_model_doc(
        """Implementation on the training stage of SnapshotEnsembleRegressor.""",  # noqa: E501
        "fit"
    )
    def fit(self,
            train_loader,
            lr_clip=None,
            epochs=100,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):
        self._validate_parameters(lr_clip, epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader,
                                                self.is_classification)

        # A dummy model used to generate snapshot ensembles
        estimator_ = self._make_estimator()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(estimator_,
                                             self.optimizer_name,
                                             **self.optimizer_args)

        scheduler = self._set_scheduler(optimizer, epochs * len(train_loader))

        # Utils
        criterion = nn.MSELoss()
        best_mse = float("inf")
        counter = 0  # a counter on generating snapshots
        n_iters_per_estimator = epochs * len(train_loader) // self.n_estimators

        # Training loop
        estimator_.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(self.device), target.to(self.device)

                # Clip the learning rate
                optimizer = self._clip_lr(optimizer, lr_clip)

                optimizer.zero_grad()
                output = estimator_(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = ("lr: {:.5f} | Epoch: {:03d} | Batch: {:03d}"
                               " | Loss: {:.5f}")
                        self.logger.info(
                            msg.format(
                                optimizer.param_groups[0]["lr"],
                                epoch,
                                batch_idx,
                                loss)
                        )

                # Snapshot ensemble updates the learning rate per iteration
                # instead of per epoch.
                scheduler.step()
                counter += 1

            if counter % n_iters_per_estimator == 0:
                # Generate and save the snapshot
                snapshot = copy.deepcopy(estimator_)
                self.estimators_.append(snapshot)

                msg = "Save the snapshot model with index: {}"
                self.logger.info(msg.format(len(self.estimators_) - 1))

            # Validation after each snapshot model being generated
            if test_loader and counter % n_iters_per_estimator == 0:
                self.eval()
                with torch.no_grad():
                    mse = 0
                    for _, (data, target) in enumerate(test_loader):
                        data = data.to(self.device)
                        target = target.to(self.device)
                        output = self.forward(data)
                        mse += criterion(output, target)
                    mse /= len(test_loader)

                    if mse < best_mse:
                        best_mse = mse
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = ("n_estimators: {} | Validation MSE: {:.5f} |"
                           " Historical Best: {:.5f}")
                    self.logger.info(
                        msg.format(
                            len(self.estimators_),
                            mse,
                            best_mse
                        )
                    )

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of SnapshotEnsembleRegressor.""",  # noqa: E501
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
