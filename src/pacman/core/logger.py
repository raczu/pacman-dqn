import logging
import logging.config
from pathlib import Path
from typing import Literal

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "[%(asctime)s] %(levelname)s: %(message)s", "datefmt": "%H:%M:%S"},
        "detailed": {
            "format": "[%(asctime)s | %(module)s] %(levelname)s: %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {"root": {"level": "INFO", "handlers": ["stdout"]}},
}


def configure_logger() -> None:
    """Configure the root logger using the predefined logging configuration."""
    logging.config.dictConfig(LOGGING_CONFIG)


def add_file_handler(
    output: Path,
    logger: logging.Logger,
    level: int = logging.INFO,
    formatter: Literal["simple", "detailed"] = "detailed",
) -> None:
    """Add a file handler to the specified logger."""
    fh = logging.FileHandler(output)
    fh.setLevel(level)
    if formatter not in LOGGING_CONFIG["formatters"]:
        raise ValueError(f"Unknown formatter: {formatter}")
    cfg = LOGGING_CONFIG["formatters"][formatter]
    fmt = logging.Formatter(fmt=cfg.get("format"), datefmt=cfg.get("datefmt"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
