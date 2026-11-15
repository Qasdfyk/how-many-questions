"""Paired tests: Wilcoxon signed-rank on per-dish APE, bootstrap CI of MAPE difference, Holm."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """CI of mean(a) - mean(b) for paired samples (percentage points if inputs are APE*100)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_test(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        p = 1.0
    else:
        p = float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    mean, lo, hi = bootstrap_diff_ci(a, b, n_boot, seed)
    return {"diff_mape": mean, "ci_lo": lo, "ci_hi": hi, "p": p}


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj.tolist()
