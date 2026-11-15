"""S3: online agent run (the VLM picks questions itself). Run after 04_precompute on the same split.

  nohup uv run python scripts/06_agent.py --split test > outputs/agent.log 2>&1 &
"""
from pbad.agent import run_agent
from pbad.runutil import base_parser, config_from_args, dishes_split

ap = base_parser(__doc__)
ap.add_argument("--split", default="test")
a = ap.parse_args()
cfg = config_from_args(a)
res = run_agent(dishes_split(cfg, a.split, a.n), cfg, a.split)
print(res.groupby(["variant", "B"]).subset.value_counts(normalize=True).groupby(level=[0, 1]).head(5).round(3))
print("invalid choice rate:", (res.n_invalid > 0).mean())
