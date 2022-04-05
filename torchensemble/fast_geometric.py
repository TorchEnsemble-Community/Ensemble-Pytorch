"""
  Motivated by geometric insights on the loss surface of deep neural networks,
  Fast Geometirc Ensembling (FGE) is an efficient ensemble that uses a
  customized learning rate scheduler to generate base estimators, similar to
  snapshot ensemble.

  Reference:
      T. Garipov, P. Izmailov, D. Podoprikhin et al., Loss Surfaces, Mode
      Connectivity, and Fast Ensembling of DNNs, NeurIPS, 2018.
"""


import torch
import logging
import warnings
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseModule, BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op
from .utils.logging import get_tb_logger


__all__ = ["FastGeometricClassifier", "FastGeometricRegressor"]


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A :mod:`DataLoader` container that contains the training data.
    cycle : int, default=4
        The number of cycles used to build each base estimator in the ensemble.
    lr_1 : float, default=5e-2
        ``alpha_1`` in original paper used to adjust the learning rate, also
        serves as the initial learning rate of the internal optimizer.
    lr_2 : float, default=1e-4
        ``alpha_2`` in original paper used to adjust the learning rate, also
        serves as the smallest learning rate of the internal optimizer.
    epochs : int, default=100
        The number of training epochs used to fit the dummy base estimator.
    log_interval : int, default=100
        The number of batches to wait before logging the training status.
    test_loader : torch.utils.data.DataLoader, default=None
        A :mod:`DataLoader` container that contains the evaluating data.

        - If ``None``, no validation is conducted after each real base
          estimator being generated.
        - If not ``None``, the ensemble will be evaluated on this
          dataloader after each base estimator being generated.
    save_model : bool, default=True
        Specify whether to save the model parameters.

        - If test_loader is ``None``, the ensemble fully trained will be
          saved.
        - If test_loader is not ``None``, the ensemble with the best
          validation performance will be saved.
    save_dir : string, default=None
        Specify where to save the model parameters.

        - If ``None``, the model will be saved in the current directory.
        - If not ``None``, the model will be saved in the specified
          directory: ``save_dir``.
"""


def _fast_geometric_model_doc(header, item="fit"):
    """
    Decorator on obtaining documentation for different fast geometric models.
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


class _BaseFastGeometric(BaseModule):
    def __init__(
        self, estimator, n_estimators, estimator_args=None, cuda=True
    ):
        super(BaseModule, self).__init__()

        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args

        if estimator_args and not isinstance(estimator, type):
            msg = (
                "The input `estimator_args` will have no effect since"
                " `estimator` is already an object after instantiation."
            )
            warnings.warn(msg, RuntimeWarning)

        self.device = torch.device("cuda" if cuda else "cpu")
        self.logger = logging.getLogger()
        self.tb_logger = get_tb_logger()

        self.estimators_ = nn.ModuleList()
        self.use_scheduler_ = False

    def _forward(self, *x):
        """
        Implementation on the internal data forwarding in fast geometric
        ensemble.
        """
        # Average
        results = [estimator(*x) for estimator in self.estimators_]
        output = op.average(results)

        return output

    def _adjust_lr(
        self, optimizer, epoch, i, n_iters, cycle, alpha_1, alpha_2
    ):
        """
        Set the internal learning rate scheduler for fast geometric ensemble.
        Please refer to the original paper for details.
        """

        def scheduler(i):
            t = ((epoch % cycle) + i) / cycle
            if t < 0.5:
                return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
            else:
                return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)

        lr = scheduler(i / n_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    @torchensemble_model_doc(
        """Set the attributes on optimizer for Fast Geometric Ensemble.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @torchensemble_model_doc(
        """Set the attributes on scheduler for Fast Geometric Ensemble.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        msg = (
            "The learning rate scheduler for fast geometirc ensemble will"
            " only be used in the first stage on building the dummy base"
            " estimator."
        )
        warnings.warn(msg, UserWarning)

        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True


@torchensemble_model_doc(
    """Implementation on the FastGeometricClassifier.""", "seq_model"
)
class FastGeometricClassifier(_BaseFastGeometric, BaseClassifier):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in FastGeometricClassifier.""",  # noqa: E501
        "classifier_forward",
    )
    def forward(self, *x):
        proba = self._forward(*x)

        return F.softmax(proba, dim=1)

    @torchensemble_model_doc(
        (
            """Set the attributes on optimizer for FastGeometricClassifier. """
            + """Notice that keyword arguments specified here will also be """
            + """used in the ensembling stage except the learning rate.."""
        ),
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name=optimizer_name, **kwargs)

    @torchensemble_model_doc(
        (
            """Set the attributes on scheduler for FastGeometricClassifier. """
            + """Notice that this scheduler will only be used in the stage on """  # noqa: E501
            + """fitting the dummy base estimator."""
        ),
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name=scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for FastGeometricClassifier.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @_fast_geometric_model_doc(
        """Implementation on the training stage of FastGeometricClassifier.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        cycle=4,
        lr_1=5e-2,
        lr_2=1e-4,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # ====================================================================
        #                Train the dummy estimator (estimator_)
        # ====================================================================

        estimator_ = self._make_estimator()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        if self.use_scheduler_:
            scheduler = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        total_iters = 0

        for epoch in range(epochs):

            # Training
            estimator_.train()
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)
                batch_size = data[0].size(0)

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        _, predicted = torch.max(output.data, 1)
                        correct = (predicted == target).sum().item()

                        msg = (
                            "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f} |"
                            " Correct: {:d}/{:d}"
                        )
                        self.logger.info(
                            msg.format(
                                epoch,
                                batch_idx,
                                loss,
                                correct,
                                batch_size,
                            )
                        )
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "fast_geometric/Base_Est/Train_Loss",
                                loss,
                                total_iters,
                            )
                total_iters += 1

            if self.use_scheduler_:
                scheduler.step()

        # ====================================================================
        #                        Generate the ensemble
        # ====================================================================

        # Set the internal optimizer
        estimator_.zero_grad()
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        # Utils
        best_acc = 0.0
        n_iters = len(train_loader)
        updated = False
        epoch = 0

        while len(self.estimators_) < self.n_estimators:

            # Training
            estimator_.train()
            for batch_idx, elem in enumerate(train_loader):

                # Update learning rate
                self._adjust_lr(
                    optimizer, epoch, batch_idx, n_iters, cycle, lr_1, lr_2
                )

                data, target = io.split_data_target(elem, self.device)
                batch_size = data[0].size(0)

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        _, predicted = torch.max(output.data, 1)
                        correct = (predicted == target).sum().item()

                        msg = (
                            "lr: {:.5f} | Epoch: {:03d} | Batch: {:03d} |"
                            " Loss: {:.5f} | Correct: {:d}/{:d}"
                        )
                        self.logger.info(
                            msg.format(
                                optimizer.param_groups[0]["lr"],
                                epoch,
                                batch_idx,
                                loss,
                                correct,
                                batch_size,
                            )
                        )
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "fast_geometric/Ensemble-Est_{}".format(
                                    len(self.estimators_)
                                )
                                + "/Train_Loss",
                                loss,
                                total_iters,
                            )
                total_iters += 1

            # Update the ensemble
            if (epoch % cycle + 1) == cycle // 2:
                base_estimator = self._make_estimator()
                base_estimator.load_state_dict(estimator_.state_dict())
                self.estimators_.append(base_estimator)
                updated = True
                total_iters = 0

                msg = "Save the base estimator with index: {}"
                self.logger.info(msg.format(len(self.estimators_) - 1))

            # Validation after each base estimator being added
            if test_loader and updated:
                self.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)
                        output = self.forward(*data)
                        _, predicted = torch.max(output.data, 1)
                        correct += (predicted == target).sum().item()
                        total += target.size(0)
                    acc = 100 * correct / total

                    if acc > best_acc:
                        best_acc = acc
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = (
                        "n_estimators: {} | Validation Acc: {:.3f} %"
                        " | Historical Best: {:.3f} %"
                    )
                    self.logger.info(
                        msg.format(len(self.estimators_), acc, best_acc)
                    )
                    if self.tb_logger:
                        self.tb_logger.add_scalar(
                            "fast_geometric/Ensemble_Est/Validation_Acc",
                            acc,
                            len(self.estimators_),
                        )
                updated = False  # reset the updating flag
            epoch += 1

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(item="classifier_evaluate")
    def evaluate(self, test_loader, return_loss=False):
        return super().evaluate(test_loader, return_loss)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)


@torchensemble_model_doc(
    """Implementation on the FastGeometricRegressor.""", "seq_model"
)
class FastGeometricRegressor(_BaseFastGeometric, BaseRegressor):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in FastGeometricRegressor.""",  # noqa: E501
        "regressor_forward",
    )
    def forward(self, *x):
        pred = self._forward(*x)
        return pred

    @torchensemble_model_doc(
        (
            """Set the attributes on optimizer for FastGeometricRegressor. """
            + """Notice that keyword arguments specified here will also be """
            + """used in the ensembling stage except the learning rate."""
        ),
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name=optimizer_name, **kwargs)

    @torchensemble_model_doc(
        (
            """Set the attributes on scheduler for FastGeometricRegressor. """
            + """Notice that this scheduler will only be used in the stage on """  # noqa: E501
            + """fitting the dummy base estimator."""
        ),
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name=scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for FastGeometricRegressor.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @_fast_geometric_model_doc(
        """Implementation on the training stage of FastGeometricRegressor.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        cycle=4,
        lr_1=5e-2,
        lr_2=1e-4,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # ====================================================================
        #                Train the dummy estimator (estimator_)
        # ====================================================================

        estimator_ = self._make_estimator()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        if self.use_scheduler_:
            scheduler = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.MSELoss()

        # Utils
        total_iters = 0

        for epoch in range(epochs):

            # Training
            estimator_.train()
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        self.logger.info(msg.format(epoch, batch_idx, loss))
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "fast_geometric/Base_Est/Train_Loss",
                                loss,
                                total_iters,
                            )
                total_iters += 1

            if self.use_scheduler_:
                scheduler.step()

        # ====================================================================
        #                        Generate the ensemble
        # ====================================================================

        # Set the internal optimizer
        estimator_.zero_grad()
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        # Utils
        best_loss = float("inf")
        n_iters = len(train_loader)
        updated = False
        epoch = 0

        while len(self.estimators_) < self.n_estimators:

            # Training
            estimator_.train()
            for batch_idx, elem in enumerate(train_loader):

                # Update learning rate
                self._adjust_lr(
                    optimizer, epoch, batch_idx, n_iters, cycle, lr_1, lr_2
                )

                data, target = io.split_data_target(elem, self.device)

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        self.logger.info(msg.format(epoch, batch_idx, loss))
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "fast_geometric/Ensemble-Est_{}".format(
                                    len(self.estimators_)
                                )
                                + "/Train_Loss",
                                loss,
                                total_iters,
                            )
                total_iters += 1

            # Update the ensemble
            if (epoch % cycle + 1) == cycle // 2:
                base_estimator = self._make_estimator()
                base_estimator.load_state_dict(estimator_.state_dict())
                self.estimators_.append(base_estimator)
                updated = True
                total_iters = 0

                msg = "Save the base estimator with index: {}"
                self.logger.info(msg.format(len(self.estimators_) - 1))

            # Validation after each base estimator being added
            if test_loader and updated:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)
                        output = self.forward(*data)
                        val_loss += self._criterion(output, target)
                    val_loss /= len(test_loader)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = (
                        "Epoch: {:03d} | Validation Loss: {:.5f} |"
                        " Historical Best: {:.5f}"
                    )
                    self.logger.info(msg.format(epoch, val_loss, best_loss))
                    if self.tb_logger:
                        self.tb_logger.add_scalar(
                            "fast_geometric/Ensemble_Est/Validation_Loss",
                            val_loss,
                            len(self.estimators_),
                        )
                updated = False  # reset the updating flag
            epoch += 1

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(item="regressor_evaluate")
    def evaluate(self, test_loader):
        return super().evaluate(test_loader)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)
