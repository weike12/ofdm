from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Union

import yaml


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML experiment config and remember its source path."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping, got {type(config).__name__}")
    config["_config_path"] = str(config_path)
    return config


def deep_get(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def merged_config(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into a copy of base."""
    result = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merged_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
