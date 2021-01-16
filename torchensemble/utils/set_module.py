import torch


def set_optimizer(model, optimizer_name, **kwargs):
    """Set the parameter optimizer for the ensemble."""
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **kwargs)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), **kwargs)
    else:
        msg = ("Unsupported type of optimizer {}, should be one of"
               " {{SGD, Adam, RMSprop}}")
        raise NotImplementedError(msg.format(optimizer_name))

    return optimizer


def set_scheduler(ensemble, scheduler_name, **kwargs):
    """Set the scheduler on learning rate for the ensemble."""
    pass
