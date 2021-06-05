import logging
from pathlib import Path
from datetime import datetime


BASE_LOG_PATH = Path(__file__).parent/'training'


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    _log_format = '%(asctime)s - [%(name)s] - %(message)s'
    FORMATTER = logging.Formatter(_log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FORMATTER)
    console_handler.setLevel(logging.DEBUG)

    log_path = BASE_LOG_PATH / '{:%Y-%m-%d-%H:%M:%S}.log'.format(datetime.now())

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

