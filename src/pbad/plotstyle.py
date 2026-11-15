"""Project-wide chart style (quiet, research-report look). Every figure in the project uses this.

Rules (see CLAUDE.md "Wykresy"):
- warm off-white surface, no box: only a hairline baseline and hairline horizontal grid, solid;
- thin marks: 2 px lines, >= 8 px markers with a 2 px surface ring, no borders around marks;
- categorical hues in fixed order, max 3 colored series per panel (validated colorblind-safe);
  anything else is de-emphasised gray; colors follow the entity (S4 is always blue, etc.);
- text is always ink (never the series color); left-aligned title + one-line takeaway subtitle;
- legend for >= 2 series, sparse direct labels (endpoints / the series the story is about);
- one y-axis only; a CSV with the plotted numbers is saved next to every figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# --- tokens (light mode; figures are for the paper/slides) --------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DEEMPH = "#b5b3ac"          # de-emphasis gray for context series

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated all-pairs (CVD dE >= 9.2)
BLUE_RAMP = {250: "#86b6ef", 450: "#2a78d6", 600: "#184f95"}

# Fixed entity -> color/marker mapping, so a strategy keeps its color in every figure.
STRATEGY_STYLE = {
    "S4":     dict(color=SERIES[0], marker="o", label="S4 uncertainty-driven (ours)", z=6),
    "S2":     dict(color=SERIES[1], marker="s", label="S2 fixed order", z=5),
    "S3":     dict(color=SERIES[2], marker="D", label="S3 VLM agent", z=4),
    "S1":     dict(color=DEEMPH, marker="o", label="S1 random", z=3),
    "oracle": dict(color=MUTED, marker="v", label="oracle (upper bound)", z=2, ls=(0, (2, 2))),
}

VARIANT_LABEL = {
    "U0": "Ideal user (U0)",
    "U1_s033": "Noisy grams, σ=0.33 (U1)",
    "U1_s050": "Noisy grams, σ=0.5 (U1)",
    "U1_s010": "Noisy grams, σ=0.1 (U1)",
    "U1_bias": "Over-estimating grams (U1, μ=0.2)",
    "U2_p010": "Category mistakes, p=0.1 (U2)",
    "U2_p020": "Category mistakes, p=0.2 (U2)",
    "U3_q020": "“Don't know” 20% (U3)",
}

QUESTION_LABEL = {
    "Q1": "Q1 weight in grams", "Q2": "Q2 small / medium / large", "Q3": "Q3 fried or oil/butter",
    "Q4": "Q4 sauce or dressing", "Q5": "Q5 main ingredient", "Q6": "Q6 list ingredients",
    "Q7": "Q7 number of items",
}

LW = 2.0          # 2 px line at 72 dpi-equivalent points
MS = 7.0          # marker diameter (pt), >= 8 px on screen
RING = 1.6        # surface ring around markers


def _font() -> str:
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "Ubuntu Sans", "Ubuntu", "Liberation Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "DejaVu Sans"


def apply() -> None:
    plt.rcParams.update({
        "font.family": _font(), "font.size": 9.5,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
        "axes.titlesize": 10.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.titlepad": 8, "axes.labelsize": 9, "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": LW, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": MS, "lines.markeredgewidth": RING, "lines.markeredgecolor": SURFACE,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def header(fig, title: str, subtitle: str | None = None, top: float = 0.985) -> None:
    """Left-aligned figure title + one-line takeaway subtitle (sentence case)."""
    fig.text(0.012, top, title, ha="left", va="top", fontsize=13, color=INK)
    if subtitle:
        fig.text(0.012, top - 0.075 * (4.0 / fig.get_figheight()), subtitle, ha="left", va="top",
                 fontsize=9.5, color=INK_2)


def strategy_line(ax, x, y, strategy: str, **kw):
    st = STRATEGY_STYLE[strategy]
    return ax.plot(x, y, color=st["color"], marker=st["marker"], ls=st.get("ls", "-"),
                   zorder=st["z"], label=st["label"], clip_on=False, **kw)[0]


def end_label(ax, x, y, text: str, dx: float = 6, color: str = INK_2, **kw):
    """Direct label at a line end; ink color, never the series color."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", ha="left",
                fontsize=8.5, color=color, annotation_clip=False, **kw)


REF_LABEL = "all 7 questions (same user)"


def ref_line(ax, y: float):
    """Horizontal reference value (e.g. 'all 7 questions'); named in the legend, not on the plot,
    so it never collides with the data lines."""
    return ax.axhline(y, color=MUTED, lw=1.0, zorder=1, label=REF_LABEL)


def save(fig, path: Path, data=None) -> None:
    """Save PNG (slides) + PDF (paper) and, if given, the plotted numbers as CSV (table view)."""
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    if data is not None:
        data.to_csv(path.with_suffix(".csv"), index=False)
    plt.close(fig)
