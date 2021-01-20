import torch
import torch.optim.lr_scheduler as lr_scheduler


def set_optimizer(model, optimizer_name, **kwargs):
    """
    Set the parameter optimizer for the model.

    Reference: https://pytorch.org/docs/stable/optim.html#algorithms
    """

    if optimizer_name == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), **kwargs)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), **kwargs)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    elif optimizer_name == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), **kwargs)
    elif optimizer_name == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), **kwargs)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), **kwargs)
    elif optimizer_name == "Rprop":
        optimizer = torch.optim.Rprop(model.parameters(), **kwargs)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **kwargs)
    else:
        msg = ("Unknown name of the optimizer {}, should be one of"
               " {{Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop,"
               " Rprop, SGD}}.")
        raise NotImplementedError(msg.format(optimizer_name))

    return optimizer


def set_scheduler(optimizer, scheduler_name, **kwargs):
    """
    Set the scheduler on learning rate for the optimizer.

    Reference:
        https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """

    if scheduler_name == "LambdaLR":
        scheduler = lr_scheduler.LambdaLR(optimizer, **kwargs)
    elif scheduler_name == "MultiplicativeLR":
        scheduler = lr_scheduler.MultiplicativeLR(optimizer, **kwargs)
    elif scheduler_name == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_name == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR(optimizer, **kwargs)
    elif scheduler_name == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR(optimizer, **kwargs)
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             **kwargs)
    else:
        msg = ("Unknown name of the scheduler {}, should be one of"
               " {{LambdaLR, MultiplicativeLR, StepLR, MultiStepLR,"
               " ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau,"
               " CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts}}.")
        raise NotImplementedError(msg.format(scheduler_name))

    return scheduler
