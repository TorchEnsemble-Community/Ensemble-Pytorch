import os
import torch


def save(model, save_dir, logger):
    """Implement model serialization to the specified directory."""
    if save_dir is None:
        save_dir = "./"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Decide the base estimator name
    if isinstance(model.base_estimator_, type):
        base_estimator_name = model.base_estimator_.__name__
    else:
        base_estimator_name = model.base_estimator_.__class__.__name__

    # {Ensemble_Model_Name}_{Base_Estimator_Name}_{n_estimators}
    filename = "{}_{}_{}_ckpt.pth".format(
        type(model).__name__,
        base_estimator_name,
        model.n_estimators,
    )

    # The real number of base estimators in some ensembles is not same as
    # `n_estimators`.
    state = {
        "n_estimators": len(model.estimators_),
        "model": model.state_dict(),
    }
    save_dir = os.path.join(save_dir, filename)

    logger.info("Saving the model to `{}`".format(save_dir))

    # Save
    torch.save(state, save_dir)

    return


def load(model, save_dir="./", logger=None):
    """Implement model deserialization from the specified directory."""
    if not os.path.exists(save_dir):
        raise FileExistsError("`{}` does not exist".format(save_dir))

    # Decide the base estimator name
    if isinstance(model.base_estimator_, type):
        base_estimator_name = model.base_estimator_.__name__
    else:
        base_estimator_name = model.base_estimator_.__class__.__name__

    # {Ensemble_Model_Name}_{Base_Estimator_Name}_{n_estimators}
    filename = "{}_{}_{}_ckpt.pth".format(
        type(model).__name__,
        base_estimator_name,
        model.n_estimators,
    )
    save_dir = os.path.join(save_dir, filename)

    if logger:
        logger.info("Loading the model from `{}`".format(save_dir))

    state = torch.load(save_dir)
    n_estimators = state["n_estimators"]
    model_params = state["model"]

    # Pre-allocate and load all base estimators
    for _ in range(n_estimators):
        model.estimators_.append(model._make_estimator())
    model.load_state_dict(model_params)


def split_data_target(element, device, logger=None):
    """Split elements in dataloader according to pre-defined rules."""
    if not isinstance(element, list):
        msg = (
            "Invalid dataloader, please check if the input dataloder is valid."
        )
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    if len(element) == 2:
        # Dataloader with one input and one target
        data, target = element[0], element[1]
        return [data.to(device)], target.to(device)  # tensor -> list
    elif len(element) > 2:
        # Dataloader with multiple inputs and one target
        data, target = element[:-1], element[-1]
        data_device = [tensor.to(device) for tensor in data]
        return data_device, target.to(device)
    else:
        # Dataloader with invalid input
        msg = (
            "The input dataloader should at least contain two tensors - data"
            " and target."
        )
        if logger:
            logger.error(msg)
        raise ValueError(msg)
