"""Figures for the paper. Style: pbad.plotstyle (project standard)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pbad import plotstyle as ps
from pbad.plotstyle import plt

ORDER = ["oracle", "S1", "S3", "S2", "S4"]   # drawing order: context first, ours on top


def budget_curves(summ: pd.DataFrame, out: Path, variants: list[str] | None = None, metric: str = "MAPE",
                  title: str = "Error vs. number of questions asked"):
    """E1/E2 headline figure: small multiples, one panel per simulated user."""
    ps.apply()
    variants = [v for v in (variants or list(summ.variant.unique())) if v in set(summ.variant)]
    n = len(variants)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols + 0.6, 2.75 * rows + 0.95),
                             sharey=True, squeeze=False)
    s0 = summ[summ.strategy == "S0"].set_index("variant")[metric]
    full = summ[summ.strategy == "full"].set_index("variant")[metric]
    plotted = []
    for ax, v in zip(axes.flat, variants):
        d = summ[summ.variant == v]
        for st in ORDER:
            dd = d[d.strategy == st].sort_values("B")
            if dd.empty:
                continue
            x, y = [0] + dd.B.tolist(), [s0.get(v)] + dd[metric].tolist()
            ps.strategy_line(ax, x, y, st, lw=ps.LW if st != "oracle" else 1.3)
            plotted += [{"variant": v, "strategy": st, "B": b, metric: val} for b, val in zip(x, y)]
        s4 = d[d.strategy == "S4"].sort_values("B")
        if len(s4):
            ps.end_label(ax, s4.B.iloc[-1], s4[metric].iloc[-1], f"{s4[metric].iloc[-1]:.0f}%")
        if v in full.index:
            ps.ref_line(ax, full[v])
            plotted.append({"variant": v, "strategy": "full", "B": 7, metric: full[v]})
        ax.set_title(ps.VARIANT_LABEL.get(v, v))
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlim(-0.15, 3.35)
        ax.set_xlabel("questions asked")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes[0, 0].set_ylabel(f"{metric}, test set (lower is better)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    names = [ps.STRATEGY_STYLE[s]["label"] for s in ["S4", "S2", "S3", "S1", "oracle"]] + [ps.REF_LABEL]
    order = [labels.index(n) for n in names if n in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
               bbox_to_anchor=(0.005, 1 - 0.62 / fig.get_figheight()), ncol=6, handlelength=1.8,
               columnspacing=1.4)
    ps.header(fig, title, "Each panel is one simulated user. Budget 0 = photo only.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / fig.get_figheight()))
    ps.save(fig, out, pd.DataFrame(plotted))


def _dumbbell(ax, labels, a, b, color_a, color_b):
    y = np.arange(len(labels))[::-1]
    for yi, ai, bi in zip(y, a, b):
        ax.plot([ai, bi], [yi, yi], color=ps.GRID, lw=2.2, zorder=1, solid_capstyle="round")
    ax.plot(a, y, "o", color=color_a, zorder=3, clip_on=False)
    ax.plot(b, y, "o", color=color_b, zorder=4, clip_on=False)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=ps.AXIS, lw=0.9, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def question_ranking(rank: pd.DataFrame, out: Path, noisy: list[str] | None = None, group: str = "all"):
    """E3/H3: single-question gain for the ideal user vs each noisy user (dumbbells)."""
    ps.apply()
    d = rank[rank.group == group]
    noisy = [v for v in (noisy or ["U1_s033", "U2_p010", "U3_q020"]) if v in set(d.variant)]
    if "U0" not in set(d.variant) or not noisy:
        return
    p = d.pivot_table(index="question", columns="variant", values="gain_pp")
    p = p.sort_values("U0", ascending=False)
    labels = [ps.QUESTION_LABEL[q] for q in p.index]
    fig, axes = plt.subplots(1, len(noisy), figsize=(3.1 * len(noisy) + 1.6, 3.6), sharey=True,
                             squeeze=False)
    for ax, v in zip(axes[0], noisy):
        _dumbbell(ax, labels, p["U0"].values, p[v].values, ps.BLUE_RAMP[250], ps.BLUE_RAMP[600])
        ax.set_title(ps.VARIANT_LABEL.get(v, v))
        ax.set_xlabel("MAPE reduction vs photo only [pp]")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=ps.BLUE_RAMP[250], label="ideal user (U0)"),
         Line2D([], [], marker="o", ls="", color=ps.BLUE_RAMP[600], label="noisy user (panel)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.62 / 3.6), ncol=2)
    ps.header(fig, "Which single question helps most, and how noise changes it",
              "One question asked (B = 1). Right of zero = the answer lowers the error.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / 3.6))
    out_df = p.reset_index()
    ps.save(fig, out, out_df)


def subgroups(rank: pd.DataFrame, out: Path, variant: str = "U0"):
    """E4/H4: single-question gain in subgroups (fat-heavy vs lean, small vs large portions)."""
    ps.apply()
    d = rank[rank.variant == variant]
    pairs = [("fat_share>30%", "fat_share<=30%", "Fat share of kcal", ">30%", "≤30%"),
             ("portion=large", "portion=small", "Portion size", "large", "small")]
    pairs = [x for x in pairs if {x[0], x[1]} <= set(d.group)]
    if not pairs:
        return
    p = d.pivot_table(index="question", columns="group", values="gain_pp")
    p = p.loc[p[pairs[0][0]].sort_values(ascending=False).index]
    labels = [ps.QUESTION_LABEL[q] for q in p.index]
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.3 * len(pairs) + 1.6, 3.6), sharey=True,
                             squeeze=False)
    from matplotlib.lines import Line2D
    for ax, (ga, gb, title, la, lb) in zip(axes[0], pairs):
        na = int(d[d.group == ga].n.iloc[0])
        nb = int(d[d.group == gb].n.iloc[0])
        _dumbbell(ax, labels, p[ga].values, p[gb].values, ps.SERIES[0], ps.SERIES[1])
        ax.set_title(title)
        ax.set_xlabel("MAPE reduction vs photo only [pp]")
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ps.SERIES[0], label=f"{la} (n={na})"),
                           Line2D([], [], marker="o", ls="", color=ps.SERIES[1], label=f"{lb} (n={nb})")],
                  loc="lower right", fontsize=8)
    ps.header(fig, "Where each question helps", f"{ps.VARIANT_LABEL.get(variant, variant)}, one question asked.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / 3.6))
    ps.save(fig, out, p.reset_index())


def all_figures(res: dict, cfg: dict):
    od = Path(res["out_dir"])
    summ, rank = res["summary"], res["ranking"]
    main = [v for v in ["U0", cfg["noise"]["main"], "U2_p010", "U3_q020"] if v in set(summ.variant)]
    budget_curves(summ, od / "fig1_budget_curves", main)
    budget_curves(summ, od / "figS1_budget_curves_all_users", None,
                  title="Error vs. number of questions asked, all simulated users")
    question_ranking(rank, od / "fig2_question_ranking")
    subgroups(rank, od / "fig3_subgroups_U0", "U0")
