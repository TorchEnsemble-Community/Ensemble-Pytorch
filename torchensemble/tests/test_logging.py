import pytest
from torchensemble.utils.logging import set_logger


def record(logger):
    logger.debug("Debug!")
    logger.info("Info!")
    logger.warning("Warning!")
    logger.error("Error!")
    logger.critical("Critical!")


def test_logger():
    logger = set_logger("Loglevel_DEBUG", "DEBUG")
    record(logger)
    logger = set_logger("Loglevel_INFO", "INFO")
    record(logger)
    logger = set_logger("Loglevel_WARNING", "WARNING")
    record(logger)
    logger = set_logger("Loglevel_ERROR", "ERROR")
    record(logger)
    logger = set_logger("Loglevel_CRITICAL", "CRITICAL")
    record(logger)

    with pytest.raises(ValueError) as excinfo:
        set_logger("Loglevel_INVALID", "INVALID")

