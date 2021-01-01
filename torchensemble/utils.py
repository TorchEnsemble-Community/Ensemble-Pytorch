import os
import time
import torch


def ctime():
    """Formatter on current time used for printing running status."""
    ctime = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())

    return ctime


def save(model, save_dir, verbose=1):
    """Implement model serialization to the specified directory."""
    if save_dir is None:
        save_dir = "./"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # {Ensemble_Method_Name}_{Base_Estimator_Name}_{n_estimators}
    filename = "{}_{}_{}_ckpt.pth".format(type(model).__name__,
                                          model.base_estimator_.__name__,
                                          model.n_estimators)
    state = {"model": model.state_dict()}
    save_dir = os.path.join(save_dir, filename)

    if verbose > 0:
        print("{} Saving the model to `{}`".format(ctime(), save_dir))

    # Save
    torch.save(state, save_dir)

    return


def set_optimizer(estimator, optimizer_name, lr, weight_decay):
    """Set the parameter optimizer for the estimator."""
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
        msg = ("Unsupported type of optimizer {}, should be one of"
               " {{SGD, Adam, RMSprop}}")
        raise NotImplementedError(msg.format(optimizer_name))

    return optimizer
