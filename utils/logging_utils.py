# -- coding: utf-8 --
import time
import datetime
import logging
import torch.distributed as dist

logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # If stream is not specified, sys.stderr is used.
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def beijing(sec):
        if time.strftime('%z') == "+0800":
            return datetime.datetime.now().timetuple()
        return (datetime.datetime.now() + datetime.timedelta(hours=8)).timetuple()

    formatter.converter = beijing

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def create_logger(name: str, file_name: str, console_level: int = logging.INFO,
                  file_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(file_name, 'w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    logger.addHandler(console_handler)

    return logger
