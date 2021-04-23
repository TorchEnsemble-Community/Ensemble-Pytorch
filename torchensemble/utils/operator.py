"""This module collects operations on tensors used in Ensemble-PyTorch."""


import torch
import numpy as np
import torch.nn.functional as F


__all__ = [
    "average",
    "sum_with_multiplicative",
    "onehot_encoding",
    "pseudo_residual_classification",
    "pseudo_residual_regression",
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


def pseudo_residual_classification(target, output, n_classes, order=1):
    """
    Compute the pseudo residual for classification with cross-entropy loss.

    Parameters
    ----------
    target :

    output :

    n_classes :

    order :

    Returns
    -------
    residual :

    """
    residual = None

    if order == 1:
        y_onehot = onehot_encoding(target, n_classes)
        residual = y_onehot - F.softmax(output, dim=1)
    else:
        n_samples = output.size(0)
        # Generate pseudo targets
        pseudo_targets = []
        for k in range(n_classes):
            y_binary = torch.ones(n_samples, 1)
            y_binary[target != k, 0] *= -1
            pseudo_targets.append(y_binary)
        pseudo_targets = torch.cat(pseudo_targets, dim=1)

    return residual


def pseudo_residual_regression(target, output, order=1):
    """Compute the pseudo residual for regression with least square error."""
    if target.size() != output.size():
        msg = "The shape of target {} should be the same as output {}."
        raise ValueError(msg.format(target.size(), output.size()))

    # The form of pesudo residuals of 1st and 2nd are the same, the parameter
    # of `order` is preserved for consistencys
    return target - output
