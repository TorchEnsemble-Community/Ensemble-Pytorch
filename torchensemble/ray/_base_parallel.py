import abc
import torch
import logging
import warnings
import torch.nn as nn


class BaseParallel(object):
    """Base class for parallel ensembles running in ray.

    WARNING: This class cannot be used directly.
    Please use the derived classes instead.
    """

    def __init__(self, estimator_provider_func, n_estimators, use_gpu=True,
                 estimator_args=None):
        super(BaseParallel, self).__init__()

        try:
            model = estimator_provider_func(**estimator_args)
            if not issubclass(model, nn.Module):
                raise RuntimeError
        except Exception as e:
            error_msg = "Cannot instantiate a local model using " \
                        "`estimator_provider_func` and `estimator_args`."
            raise RuntimeError(error_msg)

        self.estimator_provider_func = estimator_provider_func
        self.n_estimators = n_estimators
        self.use_gpu = use_gpu
        self.estimator_args = estimator_args

    def fit(self, optimizer_provider_func, loss_provider_func, train_data,
            epochs, batch_size, optimizer_args=None, loss_args=None,
            val_data=None, log_interval=100):
        """Implementation on the distributed training stage of the ensemble."""

        try:
            optimizer = optimizer_provider_func(**optimizer_args)
            if not issubclass(optimizer, torch.optim.Optimizer):
                raise RuntimeError
        except Exception as e:
            error_msg = "Cannot instantiate a local optimizer using " \
                        "`optimizer_provider_func` and `optimizer_args`."
            raise RuntimeError(error_msg)

        try:
            loss_fn = loss_provider_func(**loss_args)
            if not issubclass(loss_fn, nn.Module):
                raise RuntimeError
        except Exception as e:
            error_msg = "Cannot instantiate a loss function using " \
                       "`loss_provider_func` and `loss_args`."
            raise RuntimeError(error_msg)
