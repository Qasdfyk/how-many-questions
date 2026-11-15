"""S3: online agent. The VLM itself picks the next question from the remaining pool, budget B.

Only the choice calls are new; the final estimate for the chosen subset is the same prompt as in
precompute, so its prediction is read from the precomputed table (cache hit).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pbad.precompute import answers_for, facts_for, preds_path
from pbad.questions import POOL, QIDS, subset_key
from pbad.vlm.client import Request, VLMClient
from pbad.vlm.prompt import agent_text, parse_choice


def run_agent(df: pd.DataFrame, cfg: dict, split: str) -> pd.DataFrame:
    thr = df.attrs["q2_thresholds"]
    variants = cfg["agent"]["noise"]
    ans = answers_for(df, cfg, variants)
    client = VLMClient(cfg["vlm"], Path(cfg["paths"]["cache_dir"]))
    rows = []
    try:
        for v in variants:
            for B in cfg["agent"]["budgets"]:
                state = {d: [] for d in df.dish_id}
                invalid = {d: 0 for d in df.dish_id}
                img = dict(zip(df.dish_id, df.image))
                for step in range(B):
                    reqs = {}
                    for d, asked in state.items():
                        rem = {q: POOL[q].text for q in QIDS if q not in asked}
                        facts = facts_for(asked, ans[(d, v)], thr)
                        reqs[d] = Request(img[d], agent_text(facts, rem, step, B), 0.0)
                    outs = client.run_sync(list(reqs.values()), progress=True)
                    for (d, _), out in zip(reqs.items(), outs):
                        allowed = [q for q in QIDS if q not in state[d]]
                        q = parse_choice(out["text"], allowed)
                        if q is None:
                            invalid[d] += 1
                            q = allowed[0]
                        state[d].append(q)
                for d, asked in state.items():
                    rows.append({"dish_id": d, "variant": v, "B": B, "subset": subset_key(asked),
                                 "order": ",".join(asked), "n_invalid": invalid[d]})
    finally:
        client.close()
    res = pd.DataFrame(rows)
    res.to_parquet(preds_path(cfg, split, "agent"))
    return res
