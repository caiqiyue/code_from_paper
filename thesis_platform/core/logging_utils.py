from __future__ import annotations

import logging
from pathlib import Path


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


def setup_experiment_file_logger(
    experiment_dir: Path,
    name: str = "thesis_platform",
) -> logging.Logger:
    """Set up a file logger that writes to experiment_dir/experiment.log.

    Returns the same logger instance (with both console and file handlers).
    """

    logger = logging.getLogger(name)
    log_path = experiment_dir / "experiment.log"

    # Prevent duplicate file handlers on re-runs
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)
    return logger
