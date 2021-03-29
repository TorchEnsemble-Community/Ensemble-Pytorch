import pytest

from torchensemble.utils.logging import set_logger


def _record(logger):
    logger.debug("Debug!")
    logger.info("Info!")
    logger.warning("Warning!")
    logger.error("Error!")
    logger.critical("Critical!")


def test_logger():
    logger = set_logger("Loglevel_DEBUG", "DEBUG")
    _record(logger)
    logger = set_logger("Loglevel_INFO", "INFO")
    _record(logger)
    logger = set_logger("Loglevel_WARNING", "WARNING")
    _record(logger)
    logger = set_logger("Loglevel_ERROR", "ERROR")
    _record(logger)
    logger = set_logger("Loglevel_CRITICAL", "CRITICAL")
    _record(logger)

    with pytest.raises(ValueError) as excinfo:
        set_logger("Loglevel_INVALID", "INVALID")
    assert "INVALID" in str(excinfo.value)


def test_tb_logger():
    logger, tb_logger = set_logger(
        "Tensorboard_Logger", "DEBUG", use_tb_logger=True
    )
    _record(logger)
    if tb_logger:
        for i in range(5):
            tb_logger.add_scalar("test", 2 * i, i)
        tb_logger.close()
