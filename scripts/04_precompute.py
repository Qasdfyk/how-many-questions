"""Precompute predictions for all (dish, noise variant, subset<=3). Resumable (cache).

  uv run python scripts/04_precompute.py --split val --dry-run
  nohup uv run python scripts/04_precompute.py --split val test > outputs/precompute.log 2>&1 &
"""
from pbad.precompute import run_precompute
from pbad.runutil import base_parser, config_from_args, dishes_split

ap = base_parser(__doc__)
ap.add_argument("--split", nargs="+", default=["val", "test"])
ap.add_argument("--variants", nargs="+", default=None)
ap.add_argument("--max-budget", type=int, default=None)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
cfg = config_from_args(a)
for split in a.split:
    df = dishes_split(cfg, split, a.n)
    run_precompute(df, cfg, split, a.variants, a.max_budget, dry_run=a.dry_run)
