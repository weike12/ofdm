from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_config_snapshot(config: Mapping[str, Any], output_dir: Union[str, Path]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = {key: value for key, value in config.items() if not key.startswith("_")}
    path = output / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False, allow_unicode=True)
    return path


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    config: Mapping[str, Any],
    **extra: Any,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        **extra,
    }
    torch.save(payload, destination)


class JsonlLogger:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        serializable = {}
        for key, value in record.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
            serializable[key] = value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
