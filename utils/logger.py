import logging
import logging.handlers as handlers

from utils.structs import SingletonMeta


class Logger(metaclass=SingletonMeta):
    logger = None

    def __init__(cls, filename='log.log', name="default", level=logging.INFO):
        file_handler = handlers.RotatingFileHandler(filename)
        formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        cls.logger = logging.getLogger(name)
        cls.logger.setLevel(level)
        cls.logger.addHandler(file_handler)

    def get(cls):
        return cls.logger

    def __call__(cls):
        return cls.get()
