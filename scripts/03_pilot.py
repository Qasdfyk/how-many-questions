"""E0 pilot / F3 go-no-go for the currently served model.

Runs S0, full information (all questions, U0) and K S0 samples on N validation dishes, with and
without CoT, and reports parse-error rate, MAPE, latency and the projected cost of full precompute.

  uv run python scripts/03_pilot.py --n 10              # F2 end-to-end demo
  uv run python scripts/03_pilot.py --n 50 --cot both   # F3 pilot
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pbad.data.build import load_dishes
from pbad.precompute import model_slug, plan_requests, plan_samples
from pbad.runutil import base_parser, config_from_args, dishes_split
from pbad.vlm.client import VLMClient
from pbad.vlm.prompt import parse_estimate

ap = base_parser(__doc__)
ap.set_defaults(n=10)
ap.add_argument("--cot-mode", "--cot-both", dest="cot_mode", choices=["config", "both"], default="config")
ap.add_argument("--show", type=int, default=3, help="print N example prompts/outputs")
a = ap.parse_args()
base = config_from_args(a)
settings = [False, True] if a.cot_mode == "both" else [base["vlm"]["cot"]]

for cot in settings:
    cfg = {**base, "vlm": {**base["vlm"], "cot": cot}}
    if cot:
        cfg["vlm"]["max_tokens"] = max(cfg["vlm"]["max_tokens"], 384)
    df = dishes_split(cfg, "val", a.n)
    plan = [x for x in plan_requests(df, cfg, ["U0"]) if x[2] in ("", "ALL")]
    samples = list(plan_samples(df, cfg))
    client = VLMClient(cfg["vlm"], Path(cfg["paths"]["cache_dir"]))
    reqs = [r for *_, r in plan] + [r for *_, r in samples]
    uniq = {client.key(r): r for r in reqs}
    n_new = sum(k not in client.cache for k in uniq)
    t0 = time.perf_counter()
    client.run_sync(list(uniq.values()))
    wall = time.perf_counter() - t0
    vc = cfg["vlm"]
    gt = df.set_index("dish_id").kcal
    rows = []
    for did, v, k, r in plan:
        rec = client.cached(r)
        p = parse_estimate(rec["text"], vc["kcal_range"], vc["mass_range"])
        rows.append({"dish_id": did, "subset": k or "S0", **p, "gt_kcal": gt[did], "latency": rec["latency"],
                     "tokens_out": rec["tokens_out"], "text": rec["text"]})
    srows = [parse_estimate(client.cached(r)["text"], vc["kcal_range"], vc["mass_range"]) for *_, r in samples]
    client.close()
    res = pd.DataFrame(rows)
    res["ape"] = (res.kcal - res.gt_kcal).abs() / res.gt_kcal
    by = res.groupby("subset")
    s0_mape, full_mape = 100 * by.ape.mean()["S0"], 100 * by.ape.mean()["ALL"]

    # projected cost of full precompute: unique prompts over val+test, all variants
    full = load_dishes(cfg)
    n_uniq = 0
    for split in ("val", "test"):
        d = dishes_split(cfg, split, None)
        keys = {client.key(r) for *_, r in plan_requests(d, cfg)}
        n_uniq += len(keys) + len(d) * cfg["uncertainty"]["k"]
    sec_per_call = wall / max(n_new, 1) if n_new else float(res.latency.mean()) / vc["concurrency"]
    report = {
        "model": vc["model"], "cot": cot, "n_dishes": len(df),
        "parse_fail_rate": float((~res.ok).mean()), "clamped_rate": float(res.clamped.mean()),
        "sample_parse_fail_rate": float(np.mean([not s["ok"] for s in srows])),
        "MAPE_S0": s0_mape, "MAPE_full_U0": full_mape, "full_info_gain_pp": s0_mape - full_mape,
        "MAE_S0": float(res[res.subset == "S0"].kcal.sub(res[res.subset == "S0"].gt_kcal).abs().mean()),
        "acc20_S0": float((res[res.subset == "S0"].ape <= 0.2).mean()),
        "acc20_full": float((res[res.subset == "ALL"].ape <= 0.2).mean()),
        "mean_latency_s": float(res.latency.mean()), "mean_tokens_out": float(res.tokens_out.mean()),
        "new_calls": n_new, "wall_s": wall, "sec_per_call_throughput": sec_per_call,
        "full_precompute_unique_prompts": n_uniq,
        "full_precompute_hours_est": n_uniq * sec_per_call / 3600,
    }
    out = Path(cfg["paths"]["out_dir"]) / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{model_slug(cfg)}_n{len(df)}"
    (out / f"{tag}.json").write_text(json.dumps(report, indent=2))
    res.drop(columns=["text"]).to_csv(out / f"{tag}_per_dish.csv", index=False)
    print(json.dumps(report, indent=2))
    for _, r in res.head(a.show).iterrows():
        print(f"--- {r.dish_id} [{r.subset}] gt={r.gt_kcal:.0f} -> {r.text.strip()[:300]}")
    go = report["parse_fail_rate"] < 0.05 and 30 <= s0_mape <= 60 and full_mape < s0_mape \
        and report["full_precompute_hours_est"] <= 48
    print(f"GO/NO-GO (parse<5%, MAPE_S0 in 30-60%, full<S0, cost<=48h): {'GO' if go else 'NO-GO'}")
