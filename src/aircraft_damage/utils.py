"""Shared utilities for the aircraft damage project."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


class ProjectError(Exception):
    """Base exception for project-specific errors."""


class ConfigurationError(ProjectError):
    """Raised when configuration is invalid or incomplete."""


class AssetNotFoundError(ProjectError):
    """Raised when a required dataset, model, or image is missing."""

    def __init__(self, message: str, next_steps: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.next_steps = next_steps or []

    def __str__(self) -> str:
        if not self.next_steps:
            return self.message
        formatted_steps = "\n".join(f"- {step}" for step in self.next_steps)
        return f"{self.message}\nNext steps:\n{formatted_steps}"


def project_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[2]


def is_url(value: str) -> bool:
    """Return True when the string looks like a URL."""

    return value.startswith(("http://", "https://"))


def resolve_path(value: str | os.PathLike[str], base_dir: Path | None = None) -> Path:
    """Resolve a path relative to the project or provided base directory."""

    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if expanded.is_absolute():
        return expanded.resolve()
    base = base_dir or project_root()
    return (base / expanded).resolve()


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    """Write JSON data with stable formatting."""

    destination = Path(path)
    ensure_directory(destination.parent)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def write_text(path: str | os.PathLike[str], content: str) -> Path:
    """Write plain text content."""

    destination = Path(path)
    ensure_directory(destination.parent)
    destination.write_text(content, encoding="utf-8")
    return destination


def set_random_seed(seed: int) -> None:
    """Set random seeds across supported libraries when available."""

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
