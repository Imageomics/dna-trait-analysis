import logging


def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler("logs/default.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(process)d | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def create_child_logger(logger, child_name):
    child_logger = logger.getChild(child_name)
    handler = logging.FileHandler(f"logs/{child_name}.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    child_logger.addHandler(handler)
    return child_logger
