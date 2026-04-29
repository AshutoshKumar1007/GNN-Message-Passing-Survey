"""Structured logging that writes to both stdout and a per-run log file."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


_FORMAT = "%(asctime)s | %(levelname).1s | %(name)s | %(message)s"


def get_logger(name: str, log_dir: Path | str | None = None) -> logging.Logger:
    """Return a configured logger; idempotent on repeated calls."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def log_config(logger: logging.Logger, cfg: Any, out_path: Path | str | None = None) -> None:
    """Print and (optionally) persist the run config as JSON."""
    payload = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    text = json.dumps(payload, indent=2, default=str)
    logger.info("Run config:\n%s", text)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
