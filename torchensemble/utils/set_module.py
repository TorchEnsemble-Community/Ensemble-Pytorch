import importlib


def set_optimizer(model, optimizer_name, **kwargs):
    """
    Set the parameter optimizer for the model.

    Reference: https://pytorch.org/docs/stable/optim.html#algorithms
    """

    torch_optim_optimizers = [
        "Adadelta",
        "Adagrad",
        "Adam",
        "AdamW",
        "Adamax",
        "ASGD",
        "RMSprop",
        "Rprop",
        "SGD",
    ]
    if optimizer_name not in torch_optim_optimizers:
        msg = "Unrecognized optimizer: {}, should be one of {}."
        raise NotImplementedError(
            msg.format(optimizer_name, ",".join(torch_optim_optimizers))
        )

    optimizer_cls = getattr(
        importlib.import_module("torch.optim"), optimizer_name
    )
    optimizer = optimizer_cls(model.parameters(), **kwargs)

    return optimizer


def update_lr(optimizer, lr):
    """
    Manually update the learning rate of the optimizer. This function is used
    when the parallelization corrupts the bindings between the optimizer and
    the scheduler.
    """

    if not lr > 0:
        msg = (
            "The learning rate should be strictly positive, but got"
            " {} instead."
        )
        raise ValueError(msg.format(lr))

    for group in optimizer.param_groups:
        group["lr"] = lr

    return optimizer


def set_scheduler(optimizer, scheduler_name, **kwargs):
    """
    Set the scheduler on learning rate for the optimizer.

    Reference:
        https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """

    torch_lr_schedulers = [
        "LambdaLR",
        "MultiplicativeLR",
        "StepLR",
        "MultiStepLR",
        "ExponentialLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau",
        "CyclicLR",
        "OneCycleLR",
        "CosineAnnealingWarmRestarts",
    ]
    if scheduler_name not in torch_lr_schedulers:
        msg = "Unrecognized scheduler: {}, should be one of {}."
        raise NotImplementedError(
            msg.format(scheduler_name, ",".join(torch_lr_schedulers))
        )

    scheduler_cls = getattr(
        importlib.import_module("torch.optim.lr_scheduler"), scheduler_name
    )
    scheduler = scheduler_cls(optimizer, **kwargs)

    return scheduler
