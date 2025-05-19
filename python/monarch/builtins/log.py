import logging

from monarch.common.remote import remote


logger = logging.getLogger(__name__)


@remote(propagate="inspect")
def log_remote(*args, level: int = logging.WARNING, **kwargs) -> None:
    logger.log(level, *args, **kwargs)


@remote(propagate="inspect")
def set_logging_level_remote(level: int) -> None:
    logger.setLevel(level)
