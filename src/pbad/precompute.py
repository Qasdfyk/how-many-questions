"""Precompute VLM predictions for every dish x noise variant x question subset (|S| <= budget).

All offline strategies (S0, S1, S2, S4, oracle) are then evaluated by lookup. Identical prompts
(e.g. a noise variant that leaves the asked answers unchanged, or "I don't know" answers that reduce
to a smaller subset) share one cache entry, so they cost one VLM call.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pbad.questions import QIDS, format_fact, parse_key, subset_key, subsets
from pbad.simulator import simulate
from pbad.vlm.client import Request, VLMClient
from pbad.vlm.prompt import estimation_text, parse_estimate


def model_slug(cfg: dict) -> str:
    return cfg["vlm"]["model"].split("/")[-1] + ("_cot" if cfg["vlm"]["cot"] else "")


def preds_path(cfg: dict, split: str, kind: str = "preds") -> Path:
    return Path(cfg["paths"]["cache_dir"]) / f"{kind}_{model_slug(cfg)}_{split}.parquet"


def answers_for(df: pd.DataFrame, cfg: dict, variants: list[str] | None = None) -> dict:
    """{(dish_id, variant): {qid: answer}}"""
    thr = df.attrs["q2_thresholds"]
    vs = variants or list(cfg["noise"]["variants"])
    out = {}
    for _, d in df.iterrows():
        for v in vs:
            out[(d.dish_id, v)] = simulate(d, v, cfg["noise"]["variants"][v], thr, cfg["seed"])
    return out


def facts_for(subset, answers: dict, thr) -> list[str]:
    facts = [format_fact(q, answers[q], thr) for q in QIDS if q in subset]
    return [f for f in facts if f is not None]


def plan_requests(df: pd.DataFrame, cfg: dict, variants: list[str] | None = None,
                  max_budget: int | None = None, full_info: bool = True):
    """Yield (dish_id, variant, subset_key, Request) for all estimation calls of the protocol."""
    thr = df.attrs["q2_thresholds"]
    ans = answers_for(df, cfg, variants)
    vs = variants or list(cfg["noise"]["variants"])
    cot = cfg["vlm"]["cot"]
    for _, d in df.iterrows():
        for v in vs:
            b = min(cfg["noise"]["variants"][v].get("max_budget", cfg["questions"]["max_budget"]),
                    max_budget if max_budget is not None else 99)
            keys = [subset_key(s) for s in subsets(QIDS, b)] + (["ALL"] if full_info else [])
            a = ans[(d.dish_id, v)]
            for k in keys:
                text = estimation_text(facts_for(parse_key(k), a, thr), cot)
                yield d.dish_id, v, k, Request(d.image, text, cfg["vlm"]["temperature"])


def plan_samples(df: pd.DataFrame, cfg: dict):
    """K stochastic S0 samples per dish (uncertainty features)."""
    text = estimation_text([], cfg["vlm"]["cot"])
    for _, d in df.iterrows():
        for i in range(cfg["uncertainty"]["k"]):
            yield d.dish_id, i, Request(d.image, text, cfg["uncertainty"]["temperature"], sample=i)


def run_precompute(df: pd.DataFrame, cfg: dict, split: str, variants=None, max_budget=None,
                   chunk_dishes: int = 10, dry_run: bool = False) -> pd.DataFrame | None:
    client = VLMClient(cfg["vlm"], Path(cfg["paths"]["cache_dir"]))
    try:
        plan = list(plan_requests(df, cfg, variants, max_budget))
        samples = list(plan_samples(df, cfg))
        uniq = {client.key(r) for *_, r in plan} | {client.key(r) for *_, r in samples}
        todo = sum(k not in client.cache for k in uniq)
        print(f"[{split}] {len(df)} dishes, {len(plan)} (variant,subset) cells + {len(samples)} samples "
              f"-> {len(uniq)} unique prompts, {todo} not cached", flush=True)
        if dry_run:
            return None
        dishes = list(df.dish_id)
        for i in range(0, len(dishes), chunk_dishes):
            chunk = set(dishes[i:i + chunk_dishes])
            reqs = {client.key(r): r for did, *_, r in plan if did in chunk}
            reqs.update({client.key(r): r for did, _, r in samples if did in chunk})
            client.run_sync(list(reqs.values()), progress=False)
            print(f"  dishes {min(i + chunk_dishes, len(dishes))}/{len(dishes)}", flush=True)
        vc = cfg["vlm"]
        rows = []
        for did, v, k, r in plan:
            p = parse_estimate(client.cached(r)["text"], vc["kcal_range"], vc["mass_range"])
            rows.append({"dish_id": did, "variant": v, "subset": k, "n_q": len(parse_key(k)), **p})
        preds = pd.DataFrame(rows)
        srows = []
        for did, i, r in samples:
            p = parse_estimate(client.cached(r)["text"], vc["kcal_range"], vc["mass_range"])
            srows.append({"dish_id": did, "sample": i, **p})
        if client.failures:
            print(f"WARNING: {client.failures} VLM calls failed (stored as parse failures). "
                  "Rerun the same command to retry them.", flush=True)
        preds.to_parquet(preds_path(cfg, split))
        pd.DataFrame(srows).to_parquet(preds_path(cfg, split, "samples"))
        return preds
    finally:
        client.close()
