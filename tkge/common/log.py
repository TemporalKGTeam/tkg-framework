import datetime
import logging
import os


def setup_logger(level, echo, folder):
    # TODO(max) more readable format, maybe use tabs
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger('tkge')
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
    if level == "debug":
        return logging.DEBUG
    elif level == "warning":
        return logging.WARNING
    elif level == "error":
        return logging.ERROR
    else:
        return logging.INFO
