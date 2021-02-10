import os
import torch


def save(model, save_dir, logger):
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

    logger.info("Saving the model to `{}`".format(save_dir))

    # Save
    torch.save(state, save_dir)

    return


def load(model, save_dir="./", logger=None):
    """Implement model deserialization from the specified directory."""
    if not os.path.exists(save_dir):
        raise FileExistsError("`{}` does not exist".format(save_dir))

    # {Ensemble_Method_Name}_{Base_Estimator_Name}_{n_estimators}
    filename = "{}_{}_{}_ckpt.pth".format(type(model).__name__,
                                          model.base_estimator_.__name__,
                                          model.n_estimators)
    save_dir = os.path.join(save_dir, filename)

    if logger:
        logger.info("Loading the model from `{}`".format(save_dir))

    model.load_state_dict(torch.load(save_dir)["model"])
