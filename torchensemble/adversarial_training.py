"""
  It is known that adversatial training is able to improve the performance of
  a base estimator by treating adversarial samples as the augmented training
  data.

  Reference:
      B. Lakshminarayanan, A. Pritzel, C. Blundell., "Simple and Scalable
      Predictive Uncertainty Estimation using Deep Ensembles," NIPS 2017.
"""


import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from ._base import BaseModule, torchensemble_model_doc
from . import utils


def _parallel_fit_per_epoch(train_loader,
                            lr,
                            weight_decay,
                            epoch,
                            optimizer,
                            epsilon,
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
        data.requires_grad = True

        output = estimator(data)
        loss = criterion(output, target)
        optimizer.zero_grad()

        loss.backward()

        # Also compute the adversarial loss
        data_grad = data.grad.data
        adv_data = _get_fgsm_samples(data, epsilon, data_grad)
        adv_output = estimator(adv_data)
        adv_loss = criterion(adv_output, target)
        adv_loss.backward()

        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            # Classification
            if is_classification:
                pred = output.data.max(1)[1]
                correct = pred.eq(target.view(-1).data).sum()

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
                logger.info(msg.format(idx, epoch, batch_idx, loss))

    return estimator


def _get_fgsm_samples(sample, epsilon, sample_grad):
    """
    Private functions used to generate adversarial samples with fast gradient
    sign method (FGSM)."""
    sign_sample_grad = sample_grad.sign()
    perturbed_sample = sample + epsilon*sign_sample_grad
    perturbed_sample = torch.clamp(perturbed_sample, 0, 1)

    return perturbed_sample


class _BaseAdversarialTraining(BaseModule):

    def __init__(self,
                 estimator,
                 n_estimators,
                 estimator_args=None,
                 cuda=True,
                 n_jobs=None):
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
        self.n_jobs = n_jobs
        self.logger = logging.getLogger()

        self.estimators_ = nn.ModuleList()

    def _validate_parameters(self,
                             lr,
                             weight_decay,
                             epochs,
                             epsilon,
                             log_interval):
        """Validate hyper-parameters on training the ensemble."""

        if not lr > 0:
            msg = ("The learning rate of optimizer = {} should be strictly"
                   " positive.")
            self.logger.error(msg.format(lr))
            raise ValueError(msg.format(lr))

        if not weight_decay >= 0:
            msg = "The weight decay of optimizer = {} should not be negative."
            self.logger.error(msg.format(weight_decay))
            raise ValueError(msg.format(weight_decay))

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


class AdversarialTrainingClassifier(_BaseAdversarialTraining):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_classification = True

    @torchensemble_model_doc(
        """Implementation on the data forwarding in AdversarialTrainingClassifier.""",  # noqa: E501
        "classifier_forward")
    def forward(self, X):
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
            epsilon=0.01,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):

        # Instantiate base estimators and set attributes
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())
        self.n_outputs = self._decide_n_outputs(train_loader, True)
        self._validate_parameters(lr,
                                  weight_decay,
                                  epochs,
                                  epsilon,
                                  log_interval)

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
                        epsilon,
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

                estimators = rets  # update

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
                            self.estimators_ = nn.ModuleList()  # reset
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
        """Implementation on the evaluating stage of AdversarialTrainingClassifier.""",  # noqa: E501
        "classifier_predict")
    def predict(self, test_loader):
        self.eval()
        correct = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.forward(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()

        accuracy = 100. * float(correct) / len(test_loader.dataset)

        return accuracy


class AdversarialTrainingRegressor(_BaseAdversarialTraining):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_classification = False
        self.criterion = nn.MSELoss()

    def forward(self, X):
        pred = self._forward(X)

        return pred
