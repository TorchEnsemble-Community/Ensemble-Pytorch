"""This module collects operations on tensors used in Ensemble-PyTorch."""


import torch
import torch.nn.functional as F
from typing import List


__all__ = [
    "average",
    "sum_with_multiplicative",
    "onehot_encoding",
    "pseudo_residual_classification",
    "pseudo_residual_regression",
    "majority_vote",
]


def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)


def sum_with_multiplicative(outputs, factor):
    """
    Compuate the summation on a list of tensors, and the result is multiplied
    by a multiplicative factor.
    """
    return factor * sum(outputs)


def onehot_encoding(label, n_classes):
    """Conduct one-hot encoding on a label vector."""
    label = label.view(-1)
    onehot = torch.zeros(label.size(0), n_classes).float().to(label.device)
    onehot.scatter_(1, label.view(-1, 1), 1)

    return onehot


def pseudo_residual_classification(target, output, n_classes):
    """
    Compute the pseudo residual for classification with cross-entropyloss."""
    y_onehot = onehot_encoding(target, n_classes)

    return y_onehot - F.softmax(output, dim=1)


def pseudo_residual_regression(target, output):
    """Compute the pseudo residual for regression with least square error."""
    if target.size() != output.size():
        msg = "The shape of target {} should be the same as output {}."
        raise ValueError(msg.format(target.size(), output.size()))

    return target - output


def majority_vote(outputs: List[torch.Tensor]) -> torch.Tensor:
    """Compute the majority vote for a list of model outputs.
    outputs: list of length (n_models) of tensors with shape (n_samples, n_classes)
    majority_one_hots: (n_samples, n_classes)
    """

    assert len(outputs[0].shape) == 2, "The shape of outputs should be (n_models, n_samples, n_classes)."

    votes = torch.stack(outputs).argmax(dim=2).mode(dim=0)[0]
    proba = torch.zeros_like(outputs[0])
    majority_one_hots = proba.scatter_(1, votes.view(-1, 1), 1)

    return majority_one_hots
