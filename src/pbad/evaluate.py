"""Offline evaluation of all strategies (E1-E4) from precomputed predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pbad.precompute import answers_for, model_slug, preds_path
from pbad.questions import QIDS
from pbad.stats import holm, paired_test
from pbad.strategies import (ErrorTable, S4Selector, full_info, greedy_order, oracle, s0, s1_expected, s2,
                             selector_dataset)
from pbad.uncertainty import uncertainty_features

STRATEGIES = ["S0", "S1", "S2", "S3", "S4", "oracle"]


def _load(cfg, split, dishes):
    preds = pd.read_parquet(preds_path(cfg, split))
    samples = pd.read_parquet(preds_path(cfg, split, "samples"))
    sub = dishes[dishes.dish_id.isin(preds.dish_id.unique())].copy()
    sub.attrs = dishes.attrs
    return preds, samples, sub


def evaluate(cfg: dict, dishes: pd.DataFrame, fit_split: str = "val", eval_split: str = "test") -> dict:
    out_dir = Path(cfg["paths"]["out_dir"]) / "results" / model_slug(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_cfg = cfg["noise"]["variants"]

    pv, sv, dv = _load(cfg, fit_split, dishes)
    pt, st, dt = _load(cfg, eval_split, dishes)
    et_v, et_t = ErrorTable(pv, dv), ErrorTable(pt, dt, fallback_kcal=float(np.nanmedian(pv.kcal)))
    unc_v, unc_t = uncertainty_features(sv, pv), uncertainty_features(st, pt)
    ans_v, ans_t = answers_for(dv, cfg, et_v.variants()), answers_for(dt, cfg, et_t.variants())

    # S2 orders (per noise variant, fit on validation)
    orders = {v: greedy_order(et_v, v, et_v.max_budget(v)) for v in et_v.variants()}
    # S4 selector (pooled over variants, noise params as features, fit on validation)
    train = selector_dataset(et_v, unc_v, ans_v, noise_cfg)
    sel = S4Selector(cfg["selector"], cfg["seed"]).fit(train)
    imp = pd.Series(sel.model.feature_importances_, index=sel.features).sort_values(ascending=False)

    agent = None
    ap = preds_path(cfg, eval_split, "agent")
    if ap.exists():
        agent = pd.read_parquet(ap)

    rows = []

    def add(strategy, v, B, res: pd.DataFrame):
        r = res.reset_index().rename(columns={"index": "dish_id"})
        r["strategy"], r["variant"], r["B"] = strategy, v, B
        rows.append(r)

    for v in et_t.variants():
        maxb = et_t.max_budget(v)
        add("S0", v, 0, et_t.lookup(v, s0(et_t, v, 0)))
        if et_t.has(v, "ALL"):
            add("full", v, 7, et_t.lookup(v, full_info(et_t, v)))
        for B in range(1, maxb + 1):
            add("S1", v, B, s1_expected(et_t, v, B))
            add("S2", v, B, et_t.lookup(v, s2(et_t, v, B, orders[v])))
            add("S4", v, B, et_t.lookup(v, sel.select(et_t, unc_t, ans_t, noise_cfg, v, B)))
            add("oracle", v, B, et_t.lookup(v, oracle(et_t, v, B)))
            if agent is not None:
                a = agent[(agent.variant == v) & (agent.B == B)]
                if len(a):
                    keys = a.set_index("dish_id").subset.reindex(et_t.dishes(v)).dropna()
                    add("S3", v, B, et_t.lookup(v, keys))
    per_dish = pd.concat(rows, ignore_index=True)
    per_dish.to_parquet(out_dir / f"per_dish_{eval_split}.parquet")

    summ = summarize(per_dish, cfg["eval"]["acc_tol"])
    summ.to_csv(out_dir / f"summary_{eval_split}.csv", index=False)
    tests = significance(per_dish, cfg)
    tests.to_csv(out_dir / f"tests_{eval_split}.csv", index=False)
    ranking = question_ranking(et_t, dt)
    ranking.to_csv(out_dir / f"question_ranking_{eval_split}.csv", index=False)
    choices = (per_dish[per_dish.strategy.isin(["S3", "S4", "oracle"])]
               .groupby(["strategy", "variant", "B"]).subset.value_counts(normalize=True)
               .rename("share").reset_index())
    choices = choices[choices.share >= 0.05]
    choices.to_csv(out_dir / f"choices_{eval_split}.csv", index=False)
    imp.to_csv(out_dir / "s4_feature_importance.csv")
    meta = {"model": cfg["vlm"]["model"], "cot": cfg["vlm"]["cot"], "eval_split": eval_split,
            "n_fit": len(dv), "n_eval": len(dt), "parse_fail_rate_eval": et_t.parse_fail_rate,
            "parse_fail_rate_fit": et_v.parse_fail_rate, "s2_orders": orders,
            "agent_invalid_rate": None if agent is None else float((agent.n_invalid > 0).mean())}
    (out_dir / f"meta_{eval_split}.json").write_text(json.dumps(meta, indent=2))
    hyp = hypotheses(summ, ranking, cfg)
    (out_dir / f"hypotheses_{eval_split}.md").write_text(hyp)
    return {"per_dish": per_dish, "summary": summ, "tests": tests, "ranking": ranking,
            "meta": meta, "out_dir": out_dir, "hypotheses": hyp}


def summarize(per_dish: pd.DataFrame, tol: float) -> pd.DataFrame:
    g = per_dish.groupby(["strategy", "variant", "B"])
    s = g.agg(n=("ape", "size"), MAE=("ae", "mean"), MAPE=("ape", "mean"),
              acc20=("ape", lambda x: float((x <= tol).mean()))).reset_index()
    s["MAPE"] *= 100
    s0 = s[s.strategy == "S0"].set_index("variant").MAPE
    full_u0 = s[(s.strategy == "full") & (s.variant == "U0")].MAPE
    full_u0 = float(full_u0.iloc[0]) if len(full_u0) else np.nan
    s["recovered_gain"] = (s.variant.map(s0) - s.MAPE) / (s.variant.map(s0) - full_u0)
    # Share of the oracle's improvement at the same budget and user. Used for H1 because
    # "all questions" turned out not to be an upper bound for this VLM (F3 pilot).
    orc = s[s.strategy == "oracle"].set_index(["variant", "B"]).MAPE
    key = list(zip(s.variant, s.B))
    o = np.array([orc.get(k, np.nan) for k in key])
    s["share_of_oracle"] = (s.variant.map(s0).values - s.MAPE.values) / (s.variant.map(s0).values - o)
    # S1 is an expectation over subsets, so acc20 is not defined per dish
    s.loc[s.strategy == "S1", "acc20"] = np.nan
    return s.sort_values(["variant", "B", "strategy"]).reset_index(drop=True)


def significance(per_dish: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """H2: S4 vs S1/S2/S3 at B=1,2 for each variant; Holm within each variant."""
    rows = []
    for v, dv in per_dish.groupby("variant"):
        fam = []
        for B in (1, 2):
            a = dv[(dv.strategy == "S4") & (dv.B == B)].set_index("dish_id").ape
            if a.empty:
                continue
            for other in ("S1", "S2", "S3"):
                b = dv[(dv.strategy == other) & (dv.B == B)].set_index("dish_id").ape
                common = a.index.intersection(b.index)
                if len(common) < 5:
                    continue
                r = paired_test(a[common].values * 100, b[common].values * 100,
                                cfg["eval"]["bootstrap"], cfg["seed"])
                fam.append({"variant": v, "B": B, "comparison": f"S4 vs {other}", "n": len(common), **r})
        if fam:
            for r, adj in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = adj
            rows += fam
    return pd.DataFrame(rows)


def question_ranking(et: ErrorTable, dishes: pd.DataFrame) -> pd.DataFrame:
    """E3/E4: MAPE reduction of each single question (B=1) per variant, overall and in subgroups."""
    d = dishes.set_index("dish_id")
    groups = {
        "all": lambda idx: np.ones(len(idx), bool),
        "fat_share>10%": lambda idx: d.loc[idx, "fat_kcal_share"].values > 0.10,
        "fat_share>30%": lambda idx: d.loc[idx, "fat_kcal_share"].values > 0.30,
        "fat_share<=30%": lambda idx: d.loc[idx, "fat_kcal_share"].values <= 0.30,
        "portion=small": lambda idx: d.loc[idx, "q2_size"].values == "small",
        "portion=large": lambda idx: d.loc[idx, "q2_size"].values == "large",
        "n_ingr>=median": lambda idx: d.loc[idx, "n_ingredients"].values >= d.n_ingredients.median(),
        "n_ingr<median": lambda idx: d.loc[idx, "n_ingredients"].values < d.n_ingredients.median(),
    }
    rows = []
    for v in et.variants():
        a = et.ape.loc[v]
        for gname, fn in groups.items():
            m = fn(a.index)
            if m.sum() == 0:
                continue
            base = a.loc[m, ""].mean()
            for q in QIDS:
                if q in a.columns:
                    rows.append({"variant": v, "group": gname, "n": int(m.sum()), "question": q,
                                 "MAPE_S0": 100 * base, "MAPE_q": 100 * a.loc[m, q].mean(),
                                 "gain_pp": 100 * (base - a.loc[m, q].mean())})
    return pd.DataFrame(rows)


def hypotheses(summ: pd.DataFrame, ranking: pd.DataFrame, cfg: dict) -> str:
    main = cfg["noise"]["main"]
    L = ["# Automatic hypothesis check (thresholds from docs/03)", ""]
    s = summ.set_index(["strategy", "variant", "B"])

    def get(st, v, B, col="MAPE"):
        try:
            return float(s.loc[(st, v, B), col])
        except KeyError:
            return float("nan")

    L.append("## H1 — 2 well-chosen questions capture most of the achievable gain (U0)")
    s0u, fu = get("S0", "U0", 0), get("full", "U0", 7)
    L.append(f"- photo only {s0u:.1f}% vs all 7 questions {fu:.1f}% "
             f"({'all questions help' if fu < s0u else 'all questions do NOT help: reported as a finding'})")
    for st in ("S2", "S3", "S4", "oracle"):
        L.append(f"- {st}, B=2: MAPE {get(st, 'U0', 2):.1f}%, share of oracle gain "
                 f"{get(st, 'U0', 2, 'share_of_oracle'):.2f}, recovered vs all-7 "
                 f"{get(st, 'U0', 2, 'recovered_gain'):.2f}")
    L.append("")
    L.append("## H2 — see tests_*.csv (S4 vs S1/S2/S3, Wilcoxon + Holm)")
    L.append("")
    r = ranking[ranking.group == "all"].set_index(["variant", "question"]).gain_pp
    L.append(f"## H3 — Q1 gain drop under {main} vs U0 (threshold 30%)")
    try:
        g0, g1 = r[("U0", "Q1")], r[(main, "Q1")]
        L.append(f"- Q1 gain: U0 {g0:.1f} pp, {main} {g1:.1f} pp, drop {(1 - g1 / g0) * 100:.0f}%")
        for q in ("Q2", "Q3"):
            L.append(f"- {q} gain under {main}: {r[(main, q)]:.1f} pp (vs Q1 {g1:.1f} pp)")
        rank0 = r.loc["U0"].sort_values(ascending=False).index.tolist()
        rank1 = r.loc[main].sort_values(ascending=False).index.tolist()
        L.append(f"- ranking U0: {rank0}")
        L.append(f"- ranking {main}: {rank1}")
    except KeyError:
        L.append("- (missing data)")
    L.append("")
    L.append("## H4 — Q3 is the best single question for fat-heavy dishes (>30% kcal from fat)")
    rr = ranking[(ranking.variant == "U0") & ranking.group.isin(["fat_share>30%", "fat_share<=30%"])]
    for gname, gdf in rr.groupby("group"):
        best = gdf.sort_values("gain_pp", ascending=False).iloc[0]
        q3 = gdf[gdf.question == "Q3"].gain_pp.iloc[0]
        L.append(f"- {gname} (n={best.n}): best {best.question} ({best.gain_pp:.1f} pp), Q3 {q3:.1f} pp")
    return "\n".join(L) + "\n"
