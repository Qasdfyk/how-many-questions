"""Offline evaluation of all strategies + tests + figures (needs 04_precompute for both splits).

  uv run python scripts/05_eval.py
"""
from pbad.data.build import load_dishes
from pbad.evaluate import evaluate
from pbad.plots import all_figures
from pbad.runutil import base_parser, config_from_args

ap = base_parser(__doc__)
ap.add_argument("--fit-split", default="val")
ap.add_argument("--eval-split", default="test")
a = ap.parse_args()
cfg = config_from_args(a)
res = evaluate(cfg, load_dishes(cfg), a.fit_split, a.eval_split)
all_figures(res, cfg)
s = res["summary"]
import pandas as pd  # noqa: E402

with pd.option_context("display.width", 160, "display.max_rows", 200):
    print(s.pivot_table(index=["variant", "strategy"], columns="B", values="MAPE").round(1))
    if len(res["tests"]):
        print(res["tests"].round(4))
print(res["hypotheses"])
print("outputs in", res["out_dir"])
