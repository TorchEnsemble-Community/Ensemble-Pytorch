import torch
import pytest
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from torchensemble.utils.dataloder import FixedDataLoader


# Data
X = torch.Tensor(np.array(([0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4])))
y = torch.LongTensor(np.array(([0, 0, 1, 1])))

data = TensorDataset(X, y)
dataloder = DataLoader(data, batch_size=2, shuffle=False)


def test_fixed_dataloder():
    fixed_dataloader = FixedDataLoader(dataloder)
    for _, (fixed_elem, elem) in enumerate(zip(fixed_dataloader, dataloder)):
        # Check same elements
        for elem_1, elem_2 in zip(fixed_elem, elem):
            assert torch.equal(elem_1, elem_2)

    # Check dataloder length
    assert len(fixed_dataloader) == 2


def test_fixed_dataloader_invalid_type():
    with pytest.raises(ValueError) as excinfo:
        FixedDataLoader((X, y))
    assert "input used to instantiate FixedDataLoader" in str(excinfo.value)
