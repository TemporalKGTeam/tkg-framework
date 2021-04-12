import datetime
import logging
import os

experiment_file_handler: logging.FileHandler


def setup_logger(name, level, echo, folder, fmt):
    """
    Initializes the custom configurable logger for the whole framework.
    Use logging.getLogger(__name__) to access the logger in submodules.
    For level conventions, see https://docs.python.org/3/library/logging.html#logging-levels.
    For custom formatting, see https://docs.python.org/3/library/logging.html#logrecord-attributes.
    """
    logger = logging.getLogger(name)
    logger.setLevel(get_level(level))

    formatter = create_formatter(fmt)
    file_handler = create_file_handler(folder, formatter)
    logger.addHandler(file_handler)

    # remember experiment file handler for trial unrelated logging
    global experiment_file_handler
    experiment_file_handler = file_handler

    if echo:
        echo_handler = logging.StreamHandler()
        echo_handler.setFormatter(formatter)
        logger.addHandler(echo_handler)

    return logger


def create_formatter(fmt):
    default_fmt = "%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s (%(filename)s:%(lineno)s)"
    return logging.Formatter(fmt if fmt else default_fmt, "%Y:%m:%d %H:%M:%S")


def create_file_handler(folder, formatter):
    experiment_start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_log_file = os.path.join(folder, f"{experiment_start_time}.log")
    file_handler = logging.FileHandler(experiment_log_file)
    file_handler.setFormatter(formatter)

    return file_handler


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


def start_trial_logging(folder):
    root_logger = logging.getLogger('tkge')
    remove_all_file_handlers(root_logger)

    trial_file_handler = create_file_handler(folder, experiment_file_handler.formatter)
    root_logger.addHandler(trial_file_handler)


def stop_trial_logging():
    root_logger = logging.getLogger('tkge')
    remove_all_file_handlers(root_logger)
    root_logger.addHandler(experiment_file_handler)


def remove_all_file_handlers(logger):
    file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    for file_handler in file_handlers:
        logger.removeHandler(file_handler)
