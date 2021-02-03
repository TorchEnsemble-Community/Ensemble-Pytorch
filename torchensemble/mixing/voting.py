"""
  Extensions on the voting-based ensemble that supports heterogeneous models.
"""


import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from .._base import torchensemble_model_doc
from ..utils import io
from ..utils import set_module
from ..utils import operator as op


class MixedVotingClassifier(nn.Module):

    def __init__(self, cuda=True):
        super(MixedVotingClassifier, self).__init__()

        self.device = torch.device("cuda" if cuda else "cpu")
        self.logger = logging.getLogger()
        self.estimators_ = nn.ModuleList()
        self.is_fitted = []  # flags on whether base estimators are fitted

    def __repr__(self):
        pass

    def __str__(self):
        return self.__repr__()

    def forward(self, x):
        outputs = [F.softmax(estimator(x), dim=1)
                   for estimator in self.estimators_]
        proba = op.average(outputs)

        return proba

    def pop(self, index):
        """Pop the `index`-th base estimator in the ensemble."""
        pass

    def append(self, estimator, **estimator_args):
        """Add a unfitted base estimator into the ensemble."""
        pass

    def fit(self,
            train_loader,
            epochs=100,
            log_interval=100,
            test_loader=None,
            save_model=True,
            save_dir=None):
        """Fit the newly-added base estimator."""
        pass

    def predict(self, test_loader):
        pass
