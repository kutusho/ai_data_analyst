"""Logging utilities."""

from __future__ import annotations

import logging
from functools import lru_cache

from rich.logging import RichHandler


@lru_cache(maxsize=1)
def configure_logging() -> None:
    """Configure the application logger once."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance."""

    configure_logging()
    return logging.getLogger(name)
