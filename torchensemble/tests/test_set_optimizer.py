import pytest
import torchensemble
import torch.nn as nn


optimizer_list = ["Adadelta",
                  "Adagrad",
                  "Adam",
                  "AdamW",
                  "Adamax",
                  "ASGD",
                  "RMSprop",
                  "Rprop",
                  "SGD"]


# Base estimator
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, X):
        X = X.view(X.size()[0], -1)
        output = self.linear1(X)
        output = self.linear2(output)
        return output


@pytest.mark.parametrize("optimizer_name", optimizer_list)
def test_set_optimizer_normal(optimizer_name):
    model = MLP()
    torchensemble.utils.set_module.set_optimizer(model,
                                                 optimizer_name,
                                                 lr=1e-3)


def test_set_optimizer_abnormal():
    model = MLP()
    with pytest.raises(NotImplementedError) as excinfo:
        torchensemble.utils.set_module.set_optimizer(model, "Unknown")
    assert "Unknown name of the optimizer" in str(excinfo.value)
