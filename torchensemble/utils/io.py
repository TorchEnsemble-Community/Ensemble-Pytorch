import os
import re
import json
import yaml
import torch


def get_config(cfg_dir):
    """
    Load the json or yaml configuration file from the specified directory.
    """
    if not os.path.exists(cfg_dir):
        raise FileExistsError(
            "Configuration file does not exist: `{}`".format(cfg_dir)
        )
    _, extension = os.path.splitext(cfg_dir)

    if extension == ".yml":
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u"tag:yaml.org,2002:float",
            re.compile(
                u"""^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list(u"-+0123456789."),
        )
        with open(cfg_dir, "r") as f:
            cfg = yaml.load(f, Loader=loader)
    elif extension == ".json":
        with open(cfg_dir, "r") as f:
            cfg = json.load(f)
    else:
        msg = (
            "Unsupported type of configuration file: {}, should be one of"
            " {{yaml, json}}"
        )
        raise NotImplementedError(msg.format(extension))

    return cfg


def save(model, save_dir, logger):
    """Implement model serialization to the specified directory."""
    if save_dir is None:
        save_dir = "./"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # {Ensemble_Method_Name}_{Base_Estimator_Name}_{n_estimators}
    filename = "{}_{}_{}_ckpt.pth".format(
        type(model).__name__,
        model.base_estimator_.__name__,
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

    # {Ensemble_Method_Name}_{Base_Estimator_Name}_{n_estimators}
    filename = "{}_{}_{}_ckpt.pth".format(
        type(model).__name__,
        model.base_estimator_.__name__,
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
