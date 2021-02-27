import torch
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from torchensemble.utils import operator as op


outputs = [
    torch.FloatTensor(np.array(([1, 1], [1, 1]))),
    torch.FloatTensor(np.array(([2, 2], [2, 2]))),
    torch.FloatTensor(np.array(([3, 3], [3, 3]))),
]

label = torch.LongTensor(np.array(([0, 1, 2, 1])))
n_classes = 3


def test_average():
    actual = op.average(outputs).numpy()
    expected = np.array(([2, 2], [2, 2]))
    assert_array_equal(actual, expected)


def test_sum_with_multiplicative():
    shrinkage_rate = 0.1
    actual = op.sum_with_multiplicative(outputs, shrinkage_rate).numpy()
    expected = np.array(([0.6, 0.6], [0.6, 0.6]))
    assert_array_almost_equal(actual, expected)


def test_onehot_encoding():
    actual = op.onehot_encoding(label, n_classes).numpy()
    expected = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]))
    assert_array_almost_equal(actual, expected)


def test_residual_regression_invalid_shape():
    with pytest.raises(ValueError) as excinfo:
        op.pseudo_residual_regression(
            torch.FloatTensor(np.array(([1, 1], [1, 1]))),  # 2 * 2
            label.view(-1, 1),  # 4 * 1
        )
    assert "should be the same as output" in str(excinfo.value)
