from __future__ import annotations

import logging


def get_logger(name: str = "thesis_platform") -> logging.Logger:
    """Return a configured console logger reused across the platform."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger
