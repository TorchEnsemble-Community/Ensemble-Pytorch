import time
import pytest
from multiprocessing import Process
from torchensemble.utils.logging import MPLoggingServer, MPLoggingClient


def _record(level):
    level = level.upper()
    mp_server = MPLoggingServer("Loglevel_{}".format(level), 
                                level)
    log_server = Process(target=mp_server.log)
    log_server.start()
    
    log_client = MPLoggingClient()
    log_client.debug("Debug!")
    log_client.info("Info!")
    log_client.warn("Warn!")
    log_client.error("Error!")
    log_client.critical("Critical!")
    
    time.sleep(1)
    log_server.terminate()


def test_all_levels():
    _record("DEBUG")
    _record("INFO")
    _record("WARN")
    _record("ERROR")
    _record("CRITICAL")
