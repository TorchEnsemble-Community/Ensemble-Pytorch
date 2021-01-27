import os
import time
import logging
from multiprocessing import Queue


_level_dict = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "CRITICAL": 4}

_msg_queue = Queue()


def set_logger(log_file=None, log_console_level="info", log_file_level=None):
    """Bind the default logger with console and file stream output."""

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
            return logging.CRITICAL
        else:
            msg = ("`log_console_level` must be one of {{DEBUG, INFO,"
                   " WARNING, ERROR, CRITICAL}}, but got {} instead.")
            raise ValueError(msg.format(level.upper()))

    _logger = logging.getLogger()

    # Reset
    for h in _logger.handlers:
        _logger.removeHandler(h)

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
        print('Start logging into file {}...'.format(log_name))
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


class MPLoggingServer:
    def __init__(self, file=None, console_level="INFO", file_level="DEBUG"):
        self.file_writer = None
        if file is not None:
            log_path = os.path.join(os.getcwd(), 'logs', file + ".log")
            print("Log file will be saved in {}".format(log_path))
            if not os.path.exists(os.path.dirname(log_path)):
                os.mkdir(os.path.dirname(log_path))
            self.file_writer = open(log_path, 'wt')
        self.console_level = _level_dict[console_level.upper()]
        self.file_level = _level_dict[file_level.upper()]

    def log(self):
        while True:
            msg = _msg_queue.get()
            self._console_log(msg)
            self._file_log(msg)

    def _console_log(self, msg):
        level, info = msg
        if level >= self.console_level:
            print(info)

    def _file_log(self, msg):
        level, info = msg
        if level >= self.file_level and self.file_writer is not None:
            self.file_writer.write("{}\n".format(info))
            self.file_writer.flush()


class MPLoggingClient:
    def __init__(self):
        self.msg_queue = _msg_queue

    def _record(self, level, msg):
        time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',
                                   time.localtime(time.time()))
        self.msg_queue.put((_level_dict[level.upper()],
                            "{} [{}]:{}".format(time_stamp,
                                                level.upper(),
                                                msg)))

    def debug(self, msg):
        self._record("DEBUG", msg)

    def info(self, msg):
        self._record("INFO", msg)

    def warn(self, msg):
        self._record("WARN", msg)

    def error(self, msg):
        self._record("ERROR", msg)

    def critical(self, msg):
        self._record("CRITICAL", msg)
