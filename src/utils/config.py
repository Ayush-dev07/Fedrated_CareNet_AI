from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _dict_to_namespace(d: Any) -> Any:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d

def _namespace_to_dict(ns: Any) -> Any:
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [_namespace_to_dict(i) for i in ns]
    return ns

def load_config(path: str | Path) -> SimpleNamespace:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return SimpleNamespace()

    return _dict_to_namespace(raw)


def merge_configs(*configs: SimpleNamespace) -> SimpleNamespace:

    merged: dict[str, Any] = {}
    for cfg in configs:
        merged.update(_namespace_to_dict(cfg))
    return _dict_to_namespace(merged)


def override_from_env(cfg: SimpleNamespace, prefix: str = "FHI_") -> SimpleNamespace:

    d = _namespace_to_dict(cfg)
    for env_key, env_val in os.environ.items():
        if env_key.startswith(prefix):
            key = env_key[len(prefix):].lower()
            if key in d:
                d[key] = _cast(env_val)
    return _dict_to_namespace(d)

def _cast(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value