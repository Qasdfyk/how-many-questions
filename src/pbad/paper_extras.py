"""Extra CPU-only analyses and figures for the paper, computed from existing outputs (no VLM calls).

- dataset statistics (validation / test),
- S4 vs S2 on MAE (Wilcoxon + bootstrap) and acc@20% (McNemar exact test),
- per-dish best single question vs the fixed order (oracle headroom),
- signed bias of the model with each single question (does it over- or under-estimate?),
- pilot figure (why dishes < 50 kcal were dropped) and pipeline (method) figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from pbad import plotstyle as ps
from pbad.precompute import model_slug, preds_path
from pbad.questions import QIDS
from pbad.stats import holm, paired_test
from pbad.strategies import ErrorTable


def dataset_stats(dishes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, d in dishes.groupby("split"):
        rows.append({
            "split": split, "n": len(d),
            "kcal_mean": d.kcal.mean(), "kcal_median": d.kcal.median(),
            "kcal_min": d.kcal.min(), "kcal_max": d.kcal.max(),
            "mass_mean": d.mass_g.mean(), "mass_median": d.mass_g.median(),
            "n_ingr_median": d.n_ingredients.median(),
            "fat_share_mean": d.fat_kcal_share.mean(),
            "fat_share_gt30": (d.fat_kcal_share > 0.30).mean(),
            "has_fat_ingr": d.q3_fat.mean(), "has_sauce": d.q4_sauce.mean(),
        })
    return pd.DataFrame(rows)


def s4_vs_s2_other_metrics(per_dish: pd.DataFrame, cfg: dict, tol: float) -> pd.DataFrame:
    """MAE (Wilcoxon on per-dish AE, bootstrap CI) and acc@20% (McNemar exact) for S4 vs S2."""
    rows = []
    for v, dv in per_dish.groupby("variant"):
        fam = []
        for B in (1, 2, 3):
            a = dv[(dv.strategy == "S4") & (dv.B == B)].set_index("dish_id")
            b = dv[(dv.strategy == "S2") & (dv.B == B)].set_index("dish_id")
            common = a.index.intersection(b.index)
            if len(common) < 5:
                continue
            a, b = a.loc[common], b.loc[common]
            r = paired_test(a.ae.values, b.ae.values, cfg["eval"]["bootstrap"], cfg["seed"])
            hit_a, hit_b = a.ape.values <= tol, b.ape.values <= tol
            n10, n01 = int((hit_a & ~hit_b).sum()), int((~hit_a & hit_b).sum())
            p_mc = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            fam.append({"variant": v, "B": B, "n": len(common),
                        "MAE_S4": a.ae.mean(), "MAE_S2": b.ae.mean(), "diff_mae": r["diff_mape"],
                        "mae_ci_lo": r["ci_lo"], "mae_ci_hi": r["ci_hi"], "p_mae": r["p"],
                        "acc20_S4": hit_a.mean(), "acc20_S2": hit_b.mean(),
                        "only_S4_hit": n10, "only_S2_hit": n01, "p_acc20": float(p_mc)})
        if fam:
            for col in ("p_mae", "p_acc20"):
                for r, adj in zip(fam, holm([r[col] for r in fam])):
                    r[col + "_holm"] = adj
            rows += fam
    return pd.DataFrame(rows)


def best_single_question(et: ErrorTable, s2_first: dict[str, str]) -> pd.DataFrame:
    """Per user: how often each question is the per-dish best single question, how often that
    matches the fixed order's first question, and how often no single question beats the photo."""
    rows = []
    for v in et.variants():
        a = et.ape.loc[v]
        qs = [q for q in QIDS if q in a.columns]
        best = a[qs].idxmin(axis=1)
        no_gain = (a[qs].min(axis=1) >= a[""]).mean()
        share = best.value_counts(normalize=True)
        rows.append({"variant": v, "n": len(a), "s2_first": s2_first.get(v),
                     "best_equals_s2_first": float((best == s2_first.get(v)).mean()),
                     "no_question_helps": float(no_gain),
                     **{f"best_{q}": float(share.get(q, 0.0)) for q in qs}})
    return pd.DataFrame(rows)


def signed_bias(et: ErrorTable, v: str = "U0") -> pd.DataFrame:
    """Mean signed relative error (pred - true) / true and share of over-estimates per single question."""
    p, gt = et.pred.loc[v], et.df[et.df.variant == v].groupby("dish_id").gt_kcal.first()
    rows = []
    for key in [""] + [q for q in QIDS if q in p.columns] + (["ALL"] if "ALL" in p.columns else []):
        rel = (p[key] - gt.reindex(p.index)) / gt.reindex(p.index)
        rows.append({"variant": v, "subset": key or "S0", "mean_signed_pct": 100 * rel.mean(),
                     "median_signed_pct": 100 * rel.median(), "share_over": float((rel > 0).mean())})
    return pd.DataFrame(rows)


# --- figures ----------------------------------------------------------------------------------

def pilot_figure(per_dish_csv: Path, out: Path, min_kcal: float = 50):
    """Pilot (50 validation dishes, photo only): APE vs true kcal; near-empty plates dominate MAPE."""
    ps.apply()
    import matplotlib.pyplot as plt
    d = pd.read_csv(per_dish_csv)
    d = d[d.subset == "S0"].copy()
    d["ape_pct"] = 100 * d.ape
    small = d.gt_kcal < min_kcal
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter(d.gt_kcal[~small], d.ape_pct[~small], s=30, color=ps.SERIES[0], edgecolor=ps.SURFACE,
               linewidth=ps.RING, zorder=3, label=f"≥ {min_kcal:.0f} kcal (kept, n={int((~small).sum())})")
    ax.scatter(d.gt_kcal[small], d.ape_pct[small], s=30, color=ps.SERIES[1], edgecolor=ps.SURFACE,
               linewidth=ps.RING, zorder=3, label=f"< {min_kcal:.0f} kcal (dropped, n={int(small.sum())})")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["10", "30", "100", "300", "1000"])
    ax.set_xlabel("true energy of the dish [kcal, log scale]")
    ax.set_ylabel("absolute % error, photo only")
    m_all, m_kept = d.ape_pct.mean(), d.ape_pct[~small].mean()
    ax.legend(loc="upper right")
    ps.header(fig, "Near-empty plates dominate the percentage error",
              f"Pilot, 50 validation dishes. MAPE {m_all:.0f}% with all dishes, {m_kept:.0f}% without the "
              f"{int(small.sum())} under {min_kcal:.0f} kcal.")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    ps.save(fig, out, d[["dish_id", "gt_kcal", "kcal", "ape_pct"]])
    return {"mape_all": m_all, "mape_kept": m_kept, "n_small": int(small.sum())}


def pipeline_figure(out: Path):
    """Method diagram: photo -> question selection under budget B -> simulated user -> VLM -> kcal."""
    ps.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(5, 32)
    ax.axis("off")
    boxes = [
        (1, "Meal photo", "Nutrition5k,\noverhead RGB"),
        (21, "Question selection", "S0–S4, oracle\nbudget B ≤ 3"),
        (41, "Simulated user", "U0 ideal, U1–U3\nnoisy answers"),
        (61, "Vision-language model", "Qwen2.5-VL-7B,\nphoto + list of facts"),
        (81, "Estimate", "kcal, mass,\nconfidence (JSON)"),
    ]
    w, h, y = 18, 17, 6
    for i, (x, title, sub) in enumerate(boxes):
        fc = ps.BLUE_RAMP[450] if i == 1 else "#f1f0eb"
        tc = ps.SURFACE if i == 1 else ps.INK
        sc = "#eef4fc" if i == 1 else ps.INK_2
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=fc, ec="none"))
        ax.text(x + w / 2, y + h - 4.2, title, ha="center", va="center", fontsize=8.6, color=tc,
                weight="semibold")
        ax.text(x + w / 2, y + 5.8, sub, ha="center", va="center", fontsize=7.0, color=sc, linespacing=1.3)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + 2.6, y + h / 2), xytext=(x + w + 0.4, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=ps.MUTED, lw=1.2))
    ax.annotate("", xy=(30, y + h + 1.2), xytext=(50, y + h + 1.2),
                arrowprops=dict(arrowstyle="-|>", color=ps.MUTED, lw=1.0,
                                connectionstyle="arc3,rad=0.35"))
    ax.text(40, 29.6, "next question, conditioned on answers so far (S3, S4)", ha="center", va="center",
            fontsize=7.0, color=ps.INK_2)
    fig.tight_layout(pad=0.2)
    ps.save(fig, out)


def run_all(cfg: dict, dishes: pd.DataFrame) -> dict:
    res_dir = Path(cfg["paths"]["out_dir"]) / "results" / model_slug(cfg)
    out = res_dir / "paper"
    out.mkdir(parents=True, exist_ok=True)
    tol = cfg["eval"]["acc_tol"]

    stats = dataset_stats(dishes)
    stats.to_csv(out / "dataset_stats.csv", index=False)

    per_dish = pd.read_parquet(res_dir / "per_dish_test.parquet")
    other = s4_vs_s2_other_metrics(per_dish, cfg, tol)
    other.to_csv(out / "s4_vs_s2_mae_acc20.csv", index=False)

    meta = json.loads((res_dir / "meta_test.json").read_text())
    s2_first = {v: o[0] for v, o in meta["s2_orders"].items() if o}
    preds = pd.read_parquet(preds_path(cfg, "test"))
    dt = dishes[dishes.dish_id.isin(preds.dish_id.unique())]
    et = ErrorTable(preds, dt)
    best = best_single_question(et, s2_first)
    best.to_csv(out / "best_single_question.csv", index=False)
    bias = pd.concat([signed_bias(et, v) for v in ("U0", cfg["noise"]["main"])], ignore_index=True)
    bias.to_csv(out / "signed_bias.csv", index=False)

    pilot_csv = Path(cfg["paths"]["out_dir"]) / "pilot" / f"{model_slug(cfg)}_n50_per_dish.csv"
    pil = pilot_figure(pilot_csv, out / "fig_pilot_small_plates", cfg["data"]["min_kcal"]) \
        if pilot_csv.exists() else None
    pipeline_figure(out / "fig_pipeline")
    return {"out": out, "stats": stats, "other": other, "best": best, "bias": bias, "pilot": pil}
