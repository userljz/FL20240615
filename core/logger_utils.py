import logging
import os

logger = None

def get_logger(log_file_name, log_level=logging.DEBUG):
    global logger

    if logger is None:
        if log_file_name is None:
            raise ValueError("log_file_name cannot be None")
        
        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        logger.propagate = False  # 防止日志传播到父记录器

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # Remove all existing StreamHandlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

        # File handler
        log_file_path = os.path.join(os.getcwd(), log_file_name)
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


