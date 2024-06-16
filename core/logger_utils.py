import logging
import os

logger = None


def get_logger(log_file_name=None, log_level=logging.DEBUG):
    global logger

    if logger is None:
        if log_file_name is None:
            raise ValueError
        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)

        # Create a file handler and set the log level
        log_file_path = os.path.join(os.getcwd(), log_file_name)
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.WARNING)

        # Create a console handler and set the log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def utils_log():
    logger = get_logger('my_app.log')
    logger.error('This is an error message from utils.py')