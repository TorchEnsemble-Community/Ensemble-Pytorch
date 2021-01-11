import os
import time
import torch
import logging


def ctime():
    """Formatter on current time used for printing running status."""
    ctime = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())

    return ctime


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


def get_default_logger(log_console_level, log_file=None, log_file_level=None):
    """Bind the default logger with console and file stream output,
       where the info level is defined by user."""

    def _get_level(level):
        if level.lower() == 'debug':
            return logging.DEBUG
        elif level.lower() == 'info':
            return logging.INFO
        elif level.lower() == 'warning':
            return logging.WARN
        elif level.lower() == 'error':
            return logging.ERROR
        elif level.lower() == 'critical':
            return logging.critical
        else:
            msg = ("Param level must be a type in [DEBUG, INFO,"
                   " WARNING, ERROR, CRITICAL], but get {}")
            raise ValueError(msg.format(level.upper()))

    _logger = logging.getLogger()
    rq = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    log_path = os.path.join(os.getcwd(), 'logs')

    ch_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(_get_level(log_console_level))
    ch.setFormatter(ch_formatter)
    _logger.addHandler(ch)

    if log_file is not None:
        print('Log will be saved in \'{}\'.'.format(log_path))
        if not os.path.exists(log_path):
            os.mkdir(log_path)
            print('Create folder \'logs/\'')
        log_name = os.path.join(log_path, log_file + '-' + rq + '.log')
        print('Start logging into file {}'.format(log_name))
        fh = logging.FileHandler(log_name, mode='w')
        fh.setLevel(logging.DEBUG if log_file_level is None
                    else _get_level(log_file_level))
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - "
            "%(levelname)s: %(message)s")
        fh.setFormatter(fh_formatter)
        _logger.addHandler(fh)
    _logger.setLevel("DEBUG")
    return _logger


default_logger = get_default_logger("INFO", "ensemble-pytorch", "DEBUG")
