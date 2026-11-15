"""Shared CLI helpers for scripts/."""
from __future__ import annotations

import argparse

import pandas as pd

from pbad.config import load_config
from pbad.data.build import load_dishes


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default=None, help="YAML merged on top of configs/default.yaml")
    ap.add_argument("--model", default=None, help="override vlm.model (must match the served model)")
    ap.add_argument("--cot", choices=["on", "off"], default=None, help="override vlm.cot")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--n", type=int, default=None, help="limit to the first N dishes (seeded) of each split")
    return ap


def config_from_args(a) -> dict:
    ov: dict = {"vlm": {}}
    if a.model:
        ov["vlm"]["model"] = a.model
    if a.cot:
        ov["vlm"]["cot"] = a.cot == "on"
    if a.base_url:
        ov["vlm"]["base_url"] = a.base_url
    return load_config(a.config, ov)


def dishes_split(cfg: dict, split: str, n: int | None) -> pd.DataFrame:
    df = load_dishes(cfg)
    attrs = df.attrs
    d = df[df.split == split]
    if n is not None and n < len(d):
        d = d.sample(n=n, random_state=cfg["seed"]).sort_values("dish_id")
    d = d.reset_index(drop=True)
    d.attrs = attrs
    return d
