from __future__ import annotations

import logging
from pathlib import Path

from hierasafe_flow.utils.io import ensure_dir


def setup_logger(
    name: str = "hierasafe_flow",
    output_dir: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_dir is not None:
        ensure_dir(output_dir)
        file_handler = logging.FileHandler(Path(output_dir) / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

