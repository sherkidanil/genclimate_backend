import logging
import sys


def configure_logging(level: str | None = None) -> None:
    """Initialise root logger with sane defaults."""

    level = level or "INFO"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
