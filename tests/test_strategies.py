"""Strategies on synthetic predictions with a known optimum."""
import numpy as np
import pandas as pd

from pbad.questions import QIDS, parse_key, subset_key, subsets
from pbad.strategies import (ErrorTable, S4Selector, greedy_order, oracle, s1_expected, s2,
                             selector_dataset)
from pbad.stats import holm, paired_test
from tests.conftest import synthetic_dishes

# Q1 removes 60% of the error, Q3 30%, Q2 10%, others nothing -> optimal order Q1, Q3, Q2.
EFFECT = {"Q1": 0.6, "Q2": 0.1, "Q3": 0.3}


def synthetic_preds(dishes, variants=("U0",)):
    rows = []
    rng = np.random.default_rng(0)
    for v in variants:
        for d in dishes.itertuples():
            base_err = rng.uniform(0.3, 0.6)
            for s in subsets(QIDS, 3):
                keep = np.prod([1 - EFFECT.get(q, 0) for q in s])
                rows.append({"dish_id": d.dish_id, "variant": v, "subset": subset_key(s), "n_q": len(s),
                             "kcal": d.kcal * (1 + base_err * keep), "mass_g": d.mass_g,
                             "confidence": 0.5, "ok": True, "clamped": False})
            rows.append({"dish_id": d.dish_id, "variant": v, "subset": "ALL", "n_q": 7, "kcal": d.kcal,
                         "mass_g": d.mass_g, "confidence": 0.9, "ok": True, "clamped": False})
    return pd.DataFrame(rows)


def test_greedy_and_oracle():
    d = synthetic_dishes()
    et = ErrorTable(synthetic_preds(d), d)
    assert greedy_order(et, "U0", 3) == ["Q1", "Q3", "Q2"]
    assert set(s2(et, "U0", 2, ["Q1", "Q3", "Q2"])) == {"Q1+Q3"}
    assert set(oracle(et, "U0", 2)) == {"Q1+Q3"}
    r = s1_expected(et, "U0", 1)
    assert (r.ape > et.ape.loc["U0"]["Q1"]).all()


def test_parse_failure_falls_back_to_s0():
    d = synthetic_dishes(5)
    p = synthetic_preds(d)
    p.loc[p.subset == "Q1", ["kcal", "ok"]] = [np.nan, False]
    et = ErrorTable(p, d)
    assert np.allclose(et.pred.loc["U0"]["Q1"], et.pred.loc["U0"][""])
    assert 0 < et.parse_fail_rate < 0.05


def test_s4_learns_known_optimum(cfg):
    d = synthetic_dishes(60)
    et = ErrorTable(synthetic_preds(d), d)
    unc = pd.DataFrame({"unc_std_log_kcal": 0.1, "s0_log_kcal": np.log(et.pred.loc["U0"][""])},
                       index=d.dish_id)
    noise = {"U0": {}}
    data = selector_dataset(et, unc, {}, noise)
    assert len(data) == 60 * (7 + 7 * 6 + 21 * 5)
    sel = S4Selector({**cfg["selector"], "min_child_samples": 5}, 0).fit(data)
    chosen = sel.select(et, unc, {}, noise, "U0", 2)
    assert (chosen == "Q1+Q3").mean() > 0.9


def test_stats():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 200)
    r = paired_test(a, a + 1.0)
    assert r["p"] < 1e-6 and abs(r["diff_mape"] + 1) < 1e-9 and r["ci_hi"] < 0
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
