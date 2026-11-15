"""Minimal CLI dispatcher: python -m pbad.cli <command> [--config X] [...]."""
from __future__ import annotations

import argparse

from pbad.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["download"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.command == "download":
        from pbad.data.download import download
        download(cfg, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
