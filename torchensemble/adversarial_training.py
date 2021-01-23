"""
  Adversarial training is able to improve the performance of an ensemble by
  treating adversarial samples as the augmented training data. The fast
  gradient sign method (FGSM) is used to generate adversarial samples.

  Reference:
      B. Lakshminarayanan, A. Pritzel, C. Blundell., Simple and Scalable
      Predictive Uncertainty Estimation using Deep Ensembles, NIPS 2017.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from ._base import BaseModule, torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op


__all__ = ["_BaseAdversarialTraining",
           "AdversarialTrainingClassifier",
           "AdversarialTrainingRegressor"]


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A :mod:`DataLoader` container that contains the training data.
    epochs : int, default=100
        The number of training epochs.
    epsilon : float, defaul=0.01
        The step used to generate adversarial samples in the fast gradient
        sign method (FGSM), which should be in the range [0, 1].
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


def _adversarial_training_model_doc(header, item="fit"):
    """
    Decorator on obtaining documentation for different adversarial training
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


def _parallel_fit_per_epoch(train_loader,
                            epsilon,
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

    for batch_idx, (data, target) in enumerate(train_loader):

        batch_size = data.size()[0]
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Get adversarial samples
        _output = estimator(data)
        _loss = criterion(_output, target)
        _loss.backward()
        data_grad = data.grad.data
        adv_data = _get_fgsm_samples(data, epsilon, data_grad)

        # Compute the training loss
        optimizer.zero_grad()
        org_output = estimator(data)
        adv_output = estimator(adv_data)
        loss = criterion(org_output, target) + criterion(adv_output, target)
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            # Classification
            if is_classification:
                _, predicted = torch.max(org_output.data, 1)
                correct = (predicted == target).sum().item()

                msg = ("Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                       " | Loss: {:.5f} | Correct: {:d}/{:d}")
                print(
                    msg.format(
                        idx, epoch, batch_idx, loss, correct, batch_size
                    )
                )
            # Regression
            else:
                msg = ("Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                       " | Loss: {:.5f}")
                print(msg.format(idx, epoch, batch_idx, loss))

    return estimator


def _get_fgsm_samples(sample, epsilon, sample_grad):
    """
    Private functions used to generate adversarial samples with fast gradient
    sign method (FGSM)."""

    # Check the input range of `sample`
    min_value, max_value = torch.min(sample), torch.max(sample)
    if not 0 <= min_value < max_value <= 1:
        msg = ("The input range of samples passed to adversarial training"
               " should be in the range [0, 1], but got [{:.3f}, {:.3f}]"
               " instead.")
        raise ValueError(msg.format(min_value, max_value))

    sign_sample_grad = sample_grad.sign()
    perturbed_sample = sample + epsilon * sign_sample_grad
    perturbed_sample = torch.clamp(perturbed_sample, 0, 1)
    return perturbed_sample


class _BaseAdversarialTraining(BaseModule):

    def _validate_parameters(self, epochs, epsilon, log_interval):
        """Validate hyper-parameters on training the ensemble."""

        if not epochs > 0:
            msg = ("The number of training epochs = {} should be strictly"
                   " positive.")
            self.logger.error(msg.format(epochs))
            raise ValueError(msg.format(epochs))

        if not 0 < epsilon <= 1:
            msg = ("The step used to generate adversarial samples in FGSM"
                   " should be in the range (0, 1], but got {} instead.")
            self.logger.error(msg.format(epsilon))
            raise ValueError(msg.format(epsilon))

        if not log_interval > 0:
            msg = ("The number of batches to wait before printting the"
                   " training status should be strictly positive, but got {}"
                   " instead.")
            self.logger.error(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))


@torchensemble_model_doc("""Implementation on the AdversarialTrainingClassifier.""",  # noqa: E501
                         "model")
class AdversarialTrainingClassifier(_BaseAdversarialTraining):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_classification = True

    @torchensemble_model_doc(
        """Implementation on the data forwarding in AdversarialTrainingClassifier.""",  # noqa: E501
        "classifier_forward")
    def forward(self, x):
        # Take the average over class distributions from all base estimators.
        outputs = [F.softmax(estimator(x), dim=1)
                   for estimator in self.estimators_]
        proba = op.average(outputs)

        return proba

    @torchensemble_model_doc(
        """Set the attributes on optimizer for AdversarialTrainingClassifier.""",  # noqa: E501
        "set_optimizer")
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @torchensemble_model_doc(
        """Set the attributes on scheduler for AdversarialTrainingClassifier.""",  # noqa: E501
        "set_scheduler")
    def set_scheduler(self, scheduler_name, **kwargs):
        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True

    @_adversarial_training_model_doc(
        """Implementation on the training stage of AdversarialTrainingClassifier.""",  # noqa: E501
        "fit"
    )
    def fit(self,
            train_loader,
            epochs=100,
            epsilon=0.5,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):

        self._validate_parameters(epochs, epsilon, log_interval)
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
                        epsilon,
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

                estimators = rets  # update

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
                            self.estimators_ = nn.ModuleList()  # reset
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
        self.estimators_.extend(rets)
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of AdversarialTrainingClassifier.""",  # noqa: E501
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


@torchensemble_model_doc("""Implementation on the AdversarialTrainingRegressor.""",  # noqa: E501
                         "model")
class AdversarialTrainingRegressor(_BaseAdversarialTraining):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_classification = False

    @torchensemble_model_doc(
        """Implementation on the data forwarding in AdversarialTrainingRegressor.""",  # noqa: E501
        "regressor_forward")
    def forward(self, x):
        # Take the average over predictions from all base estimators.
        outputs = [estimator(x) for estimator in self.estimators_]
        pred = op.average(outputs)

        return pred

    @torchensemble_model_doc(
        """Set the attributes on optimizer for AdversarialTrainingRegressor.""",  # noqa: E501
        "set_optimizer")
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @torchensemble_model_doc(
        """Set the attributes on scheduler for AdversarialTrainingRegressor.""",  # noqa: E501
        "set_scheduler")
    def set_scheduler(self, scheduler_name, **kwargs):
        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True

    @_adversarial_training_model_doc(
        """Implementation on the training stage of AdversarialTrainingRegressor.""",  # noqa: E501
        "fit"
    )
    def fit(self,
            train_loader,
            epochs=100,
            epsilon=0.5,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):

        self._validate_parameters(epochs, epsilon, log_interval)
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
                        epsilon,
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

                estimators = rets  # update

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
        self.estimators_.extend(rets)
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of AdversarialTrainingRegressor.""",  # noqa: E501
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
