"""Extra CPU-only analyses and figures for the paper (needs 05_eval output; no VLM calls).

  uv run python scripts/07_paper_extras.py
"""
import pandas as pd

from pbad.data.build import load_dishes
from pbad.paper_extras import run_all
from pbad.runutil import base_parser, config_from_args

a = base_parser(__doc__).parse_args()
cfg = config_from_args(a)
r = run_all(cfg, load_dishes(cfg))
with pd.option_context("display.width", 200, "display.max_columns", 30):
    for k in ("stats", "other", "best", "bias"):
        print(f"\n== {k}")
        print(r[k].round(3).to_string(index=False))
print("\npilot:", r["pilot"])
print("outputs in", r["out"])
