"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .utils import ConfigurationError, is_url, project_root, resolve_path

PATH_HINTS = ("path", "dir", "root", "file", "checkpoint", "archive")


def _looks_like_path(key: str) -> bool:
    return any(hint in key.lower() for hint in PATH_HINTS)


def _resolve_paths(value: Any, base_dir: Path, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_paths(item, base_dir=base_dir, parent_key=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_paths(item, base_dir=base_dir, parent_key=parent_key) for item in value
        ]
    if isinstance(value, str) and _looks_like_path(parent_key) and not is_url(value):
        return str(resolve_path(value, base_dir=base_dir))
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and resolve relative paths."""

    config_path = resolve_path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Configuration file must contain a YAML mapping at the top level: {config_path}"
        )

    config = _resolve_paths(raw, base_dir=project_root())
    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(project_root()),
    }
    return config
