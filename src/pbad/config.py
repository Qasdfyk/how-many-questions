from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load YAML config (default: configs/default.yaml), optionally merging another file on top."""
    with open(ROOT / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != (ROOT / "configs" / "default.yaml").resolve():
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        _deep_update(cfg, overrides)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else ROOT / p
    return cfg


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
