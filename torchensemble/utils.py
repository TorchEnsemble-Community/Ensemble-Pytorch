import torch

def set_optimizer(estimator, optimizer_name, lr, weight_decay):
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(estimator.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(estimator.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(estimator.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay)
    else:
        msg = "Unsupported type of optimizer {}"
        raise NotImplementedError(msg.format(optimizer_name))

    return optimizer
