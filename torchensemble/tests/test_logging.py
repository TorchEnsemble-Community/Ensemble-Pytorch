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
