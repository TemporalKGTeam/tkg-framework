import datetime
import logging
import os


def setup_logger(name, level, echo, folder, fmt):
    """
    Initializes the custom configurable logger for the whole framework.
    Use logging.getLogger(__name__) to access the logger in submodules.
    For level conventions, see https://docs.python.org/3/library/logging.html#logging-levels.
    For custom formatting, see https://docs.python.org/3/library/logging.html#logrecord-attributes.
    """
    default_fmt = "%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s (%(filename)s:%(lineno)s)"
    formatter = logging.Formatter(fmt if fmt else default_fmt, "%Y:%m:%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(get_level(level))

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = os.path.join(folder, f"{start_time}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if echo:
        echo_handler = logging.StreamHandler()
        echo_handler.setFormatter(formatter)
        logger.addHandler(echo_handler)

    return logger


def get_level(level):
    if level == "notset":
        return logging.NOTSET
    elif level == "debug":
        return logging.DEBUG
    elif level == "warning":
        return logging.WARNING
    elif level == "error":
        return logging.ERROR
    elif level == "critical":
        return logging.CRITICAL
    else:
        return logging.INFO
