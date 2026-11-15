"""Question-selection strategies evaluated offline on precomputed predictions.

Each strategy maps (variant, budget B, dish) -> asked subset (canonical key). S1 (random) is
evaluated exactly as the expectation over all subsets of size B.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from pbad.questions import QIDS, subset_key

NUM_FEATURES_NOISE = ("mass_sigma", "mass_mu", "cat_p", "dont_know_q")


class ErrorTable:
    """Per (variant, dish, subset) prediction and errors, with S0 fallback for unparseable outputs."""

    def __init__(self, preds: pd.DataFrame, dishes: pd.DataFrame, fallback_kcal: float | None = None):
        gt = dishes.set_index("dish_id").kcal
        p = preds.copy()
        s0 = p[p.subset == ""].set_index(["variant", "dish_id"]).kcal
        miss = p.kcal.isna()
        p.loc[miss, "kcal"] = [s0.get((v, d), np.nan) for v, d in zip(p.variant[miss], p.dish_id[miss])]
        fb = fallback_kcal if fallback_kcal is not None else float(np.nanmedian(p.kcal))
        p["kcal"] = p.kcal.fillna(fb)
        p["gt_kcal"] = p.dish_id.map(gt)
        p["ae"] = (p.kcal - p.gt_kcal).abs()
        p["ape"] = p.ae / p.gt_kcal
        self.df = p
        self.ape = p.pivot_table(index=["variant", "dish_id"], columns="subset", values="ape")
        self.ae = p.pivot_table(index=["variant", "dish_id"], columns="subset", values="ae")
        self.pred = p.pivot_table(index=["variant", "dish_id"], columns="subset", values="kcal")
        self.conf = p.pivot_table(index=["variant", "dish_id"], columns="subset", values="confidence")
        self.mass = p.pivot_table(index=["variant", "dish_id"], columns="subset", values="mass_g")
        self.parse_fail_rate = float(preds.ok.eq(False).mean())

    def variants(self) -> list[str]:
        return list(self.ape.index.get_level_values(0).unique())

    def dishes(self, v: str) -> list[str]:
        return list(self.ape.loc[v].index)

    def has(self, v: str, key: str) -> bool:
        return key in self.ape.columns and self.ape.loc[v, key].notna().all()

    def max_budget(self, v: str) -> int:
        cols = [c for c in self.ape.columns if c != "ALL" and self.ape.loc[v, c].notna().all()]
        return max(len(c.split("+")) if c else 0 for c in cols)

    def lookup(self, v: str, keys: pd.Series) -> pd.DataFrame:
        """keys: dish_id -> subset key. Returns per-dish pred/ae/ape."""
        a, e, pr = self.ape.loc[v], self.ae.loc[v], self.pred.loc[v]
        idx = keys.index
        cols = [a.columns.get_loc(k) for k in keys.values]
        rows = [a.index.get_loc(d) for d in idx]
        return pd.DataFrame({"pred": pr.values[rows, cols], "ae": e.values[rows, cols],
                             "ape": a.values[rows, cols], "subset": keys.values}, index=idx)


# --- simple strategies -------------------------------------------------------------------------

def s0(et: ErrorTable, v: str, B: int) -> pd.Series:
    return pd.Series("", index=et.dishes(v))


def s1_expected(et: ErrorTable, v: str, B: int) -> pd.DataFrame:
    keys = [subset_key(c) for c in combinations(QIDS, B)]
    return pd.DataFrame({"pred": np.nan, "ae": et.ae.loc[v][keys].mean(axis=1),
                         "ape": et.ape.loc[v][keys].mean(axis=1), "subset": f"random{B}"})


def greedy_order(et: ErrorTable, v: str, max_b: int) -> list[str]:
    """S2: global greedy order minimising mean APE (fit on validation)."""
    order: list[str] = []
    for _ in range(max_b):
        best = min((q for q in QIDS if q not in order),
                   key=lambda q: et.ape.loc[v][subset_key(order + [q])].mean())
        order.append(best)
    return order


def s2(et: ErrorTable, v: str, B: int, order: list[str]) -> pd.Series:
    return pd.Series(subset_key(order[:B]), index=et.dishes(v))


def oracle(et: ErrorTable, v: str, B: int) -> pd.Series:
    keys = [subset_key(c) for k in range(B + 1) for c in combinations(QIDS, k)]
    return et.ape.loc[v][keys].idxmin(axis=1)


def full_info(et: ErrorTable, v: str) -> pd.Series:
    return pd.Series("ALL", index=et.dishes(v))


# --- S4: uncertainty-driven selector ----------------------------------------------------------

def _answer_feats(ans: dict, S) -> dict:
    f = {}
    a = ans or {}
    f["ans_q1_logmass"] = np.log(a["Q1"]) if "Q1" in S and a.get("Q1") is not None else np.nan
    f["ans_q2_size"] = ({"small": 0, "medium": 1, "large": 2}[a["Q2"]]
                        if "Q2" in S and a.get("Q2") is not None else np.nan)
    f["ans_q3_fat"] = float(a["Q3"]) if "Q3" in S and a.get("Q3") is not None else np.nan
    f["ans_q4_sauce"] = float(a["Q4"]) if "Q4" in S and a.get("Q4") is not None else np.nan
    f["ans_q6_n"] = float(len(a["Q6"])) if "Q6" in S and a.get("Q6") is not None else np.nan
    f["ans_q7_n"] = float(a["Q7"]) if "Q7" in S and a.get("Q7") is not None else np.nan
    return f


def selector_rows(et: ErrorTable, unc: pd.DataFrame, answers: dict, noise_cfg: dict,
                  v: str, d: str, S: tuple[str, ...], with_target: bool) -> list[dict]:
    key = subset_key(S)
    base = {**unc.loc[d].to_dict(),
            **{k: noise_cfg[v].get(k, 0.0) for k in NUM_FEATURES_NOISE},
            "cur_log_kcal": np.log(et.pred.loc[(v, d), key]),
            "cur_conf": et.conf.loc[(v, d), key],
            "cur_log_mass": np.log(et.mass.loc[(v, d), key]) if et.mass.loc[(v, d), key] > 0 else np.nan,
            "n_asked": len(S),
            **{f"in_{q}": float(q in S) for q in QIDS},
            **_answer_feats(answers.get((d, v)), S)}
    base["cur_shift"] = base["cur_log_kcal"] - base["s0_log_kcal"]
    rows = []
    for i, q in enumerate(QIDS):
        if q in S:
            continue
        nk = subset_key(S + (q,))
        if nk not in et.ape.columns or np.isnan(et.ape.loc[(v, d), nk]):
            continue
        r = {**base, "cand": i, "dish_id": d, "variant": v, "S": key, "q": q}
        if with_target:
            r["gain"] = et.ape.loc[(v, d), key] - et.ape.loc[(v, d), nk]
        rows.append(r)
    return rows


def selector_dataset(et: ErrorTable, unc: pd.DataFrame, answers: dict, noise_cfg: dict,
                     max_b: int = 3) -> pd.DataFrame:
    rows = []
    for v in et.variants():
        b = min(et.max_budget(v), max_b)
        for d in et.dishes(v):
            for k in range(b):
                for S in combinations(QIDS, k):
                    rows += selector_rows(et, unc, answers, noise_cfg, v, d, S, True)
    return pd.DataFrame(rows)


META = ("dish_id", "variant", "S", "q", "gain")


class S4Selector:
    def __init__(self, scfg: dict, seed: int):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(
            n_estimators=scfg["n_estimators"], learning_rate=scfg["learning_rate"],
            num_leaves=scfg["num_leaves"], min_child_samples=scfg["min_child_samples"],
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="huber", alpha=0.3,
            random_state=seed, verbose=-1)
        self.features: list[str] = []

    def fit(self, data: pd.DataFrame) -> "S4Selector":
        self.features = [c for c in data.columns if c not in META]
        self.model.fit(data[self.features], data["gain"], categorical_feature=["cand"])
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(rows[self.features])

    def select(self, et: ErrorTable, unc, answers, noise_cfg, v: str, B: int) -> pd.Series:
        out = {}
        for d in et.dishes(v):
            S: tuple[str, ...] = ()
            for _ in range(B):
                rows = pd.DataFrame(selector_rows(et, unc, answers, noise_cfg, v, d, S, False))
                if rows.empty:
                    break
                q = rows.q.values[int(np.argmax(self.predict(rows)))]
                S = tuple(x for x in QIDS if x in S + (q,))
            out[d] = subset_key(S)
        return pd.Series(out)
