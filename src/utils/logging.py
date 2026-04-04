from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_loggers: set[str] = set()

def get_logger(
    name: str,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger

def set_global_level(level: int | str) -> None:
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    for name in _configured_loggers:
        logging.getLogger(name).setLevel(level)