import os
import time
import logging


_tb_logger = None
__all__ = ["set_logger", "get_tb_logger"]


def set_logger(
    log_file=None,
    log_console_level="info",
    log_file_level=None,
    use_tb_logger=False,
):
    """Bind the default logger with console and file stream output."""

    def _get_level(level):
        if level.lower() == "debug":
            return logging.DEBUG
        elif level.lower() == "info":
            return logging.INFO
        elif level.lower() == "warning":
            return logging.WARN
        elif level.lower() == "error":
            return logging.ERROR
        elif level.lower() == "critical":
            return logging.CRITICAL
        else:
            msg = (
                "`log_console_level` must be one of {{DEBUG, INFO,"
                " WARNING, ERROR, CRITICAL}}, but got {} instead."
            )
            raise ValueError(msg.format(level.upper()))

    _logger = logging.getLogger()

    # Reset
    for h in _logger.handlers:
        _logger.removeHandler(h)

    rq = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
    log_path = os.path.join(os.getcwd(), "logs")

    ch_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(_get_level(log_console_level))
    ch.setFormatter(ch_formatter)
    _logger.addHandler(ch)

    if log_file is not None:
        print("Log will be saved in '{}'.".format(log_path))
        if not os.path.exists(log_path):
            os.mkdir(log_path)
            print("Create folder 'logs/'")
        log_name = os.path.join(log_path, log_file + "-" + rq + ".log")
        print("Start logging into file {}...".format(log_name))
        fh = logging.FileHandler(log_name, mode="w")
        fh.setLevel(
            logging.DEBUG
            if log_file_level is None
            else _get_level(log_file_level)
        )
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - "
            "%(levelname)s: %(message)s"
        )
        fh.setFormatter(fh_formatter)
        _logger.addHandler(fh)
    _logger.setLevel("DEBUG")

    if use_tb_logger:
        tb_log_path = os.path.join(
            log_path, log_file + "-" + rq + "_tb_logger"
        )
        os.mkdir(tb_log_path)
        init_tb_logger(log_dir=tb_log_path)

    return _logger


def init_tb_logger(log_dir):
    try:
        import tensorboard  # noqa: F401
    except ModuleNotFoundError:
        msg = (
            "Cannot load the module tensorboard. Please make sure that"
            " tensorboard is installed."
        )
        raise ModuleNotFoundError(msg)

    from torch.utils.tensorboard import SummaryWriter

    global _tb_logger

    if not _tb_logger:
        _tb_logger = SummaryWriter(log_dir=log_dir)


def get_tb_logger():
    return _tb_logger
