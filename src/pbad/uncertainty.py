"""Uncertainty features per dish from K stochastic S0 samples + the greedy S0 prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def uncertainty_features(samples: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    s = samples[samples.ok].copy()
    s["log_kcal"] = np.log(s.kcal)
    g = s.groupby("dish_id")
    feats = pd.DataFrame({
        "unc_std_log_kcal": g.log_kcal.std(),
        "unc_mean_log_kcal": g.log_kcal.mean(),
        "unc_cv_kcal": g.kcal.std() / g.kcal.mean(),
        "unc_mean_conf": g.confidence.mean(),
        "unc_std_log_mass": g.mass_g.apply(lambda x: np.log(x.dropna()).std()),
    })
    s0 = preds[(preds.variant == "U0") & (preds.subset == "")].set_index("dish_id")
    feats["s0_log_kcal"] = np.log(s0.kcal)
    feats["s0_log_mass"] = np.log(s0.mass_g)
    feats["s0_conf"] = s0.confidence
    feats["s0_kcal_density"] = s0.kcal / s0.mass_g
    return feats
