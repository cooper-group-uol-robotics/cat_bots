"""Figures for benchmark campaigns (static PNG via matplotlib).

Colours follow the entity, never its rank, using the validated categorical order
(blue, orange, aqua, yellow, magenta). Three of them sit below 3:1 contrast on the light surface,
so every line is direct-labelled as well as listed in the legend, and every figure is written
together with a CSV of the plotted numbers.
"""

from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# fixed entity -> colour assignment, shared by every figure
COLOURS = {
    "pipeline": SLOTS[0], "pipeline (Kraken + RDKit)": SLOTS[0],
    "random": SLOTS[1], "bo": SLOTS[2], "BO": SLOTS[2],
    "pipeline (CheMeleon)": SLOTS[3], "pipeline (combined)": SLOTS[4], "gbo": SLOTS[3],
}
LABELS = {"pipeline": "Overview pipeline", "random": "Random", "bo": "Plain BO", "gbo": "Generality BO"}


def _style(ax, xlabel: str, ylabel: str, title: str, title_y: float = 1.0):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10, y=title_y)


def _mean_ci(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% CI (normal approx.) per column, ignoring NaN (runs that ended earlier)."""
    n = np.sum(~np.isnan(stack), axis=0)
    mean = np.nanmean(stack, axis=0)
    se = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _stage_bands(ax, spans: Sequence[tuple[str, float, float]]) -> None:
    """Shade alternate stage spans and label them on two staggered rows above the plot."""
    for k, (stage, start, end) in enumerate(spans):
        if k % 2 == 0:
            ax.axvspan(start, end, color=GRID, alpha=0.45, linewidth=0)
        ax.annotate(stage, ((start + end) / 2, 1.0 + 0.055 * (k % 2)), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=8, color=INK_2, annotation_clip=False)


def _legend_below(ax, ncol: int, offset: float = -0.16):
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol)


def _direct_labels(ax, ends: list[tuple[float, float, str, str]], min_gap_frac: float = 0.06):
    """Line-end labels, nudged vertically so they never overlap (keeps their order)."""
    lo, hi = ax.get_ylim()
    gap = min_gap_frac * (hi - lo)
    placed = []
    for x, y, text, colour in sorted(ends, key=lambda e: e[1]):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(text, (x, y), xytext=(x, y_text), textcoords="data", va="center", fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=colour, lw=1.2),
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.8) if y_text != y else None,
                    annotation_clip=False)


def plot_curves(curves: Mapping[str, Sequence[np.ndarray]], path, title: str, ylabel: str,
                xlabel: str = "Experiment number", reference_lines: Mapping[str, float] | None = None,
                stage_bounds: Mapping[str, float] | None = None, x: Sequence[int] | None = None) -> pd.DataFrame:
    """Mean ± 95% CI per strategy of per-run curves (ragged runs padded with NaN).

    ``x``: shared x positions (default 1..max length). ``stage_bounds``: label -> x where a stage ends.
    Returns (and writes next to ``path`` as CSV) the plotted table.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    table, ends = {}, []
    for name, runs in curves.items():
        length = max(len(r) for r in runs)
        stack = np.full((len(runs), length), np.nan)
        for i, r in enumerate(runs):
            stack[i, : len(r)] = r
        xs = np.asarray(x[:length] if x is not None else np.arange(1, length + 1))
        keep = np.sum(~np.isnan(stack), axis=0) >= 0.5 * len(runs)  # only where most runs contribute
        mean, lo, hi = _mean_ci(stack)
        lo = np.maximum(lo, 0.0)  # yields cannot be negative
        colour = COLOURS.get(name, SLOTS[len(ends) % len(SLOTS)])
        label = LABELS.get(name, name)
        ax.fill_between(xs[keep], lo[keep], hi[keep], color=colour, alpha=0.15, linewidth=0)
        ax.plot(xs[keep], mean[keep], color=colour, linewidth=2, label=label)
        ends.append((xs[keep][-1], mean[keep][-1], label, colour))
        table[f"{label} mean"] = pd.Series(mean[keep], index=xs[keep])
        table[f"{label} ci_low"] = pd.Series(lo[keep], index=xs[keep])
        table[f"{label} ci_high"] = pd.Series(hi[keep], index=xs[keep])
    for text, yv in (reference_lines or {}).items():
        ax.axhline(yv, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(text, (0, yv), xycoords=("axes fraction", "data"), xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=INK_2)
    spans, prev = [], 0.0
    for stage, end in (stage_bounds or {}).items():
        spans.append((stage, prev, end))
        prev = end
    _stage_bands(ax, spans)
    _style(ax, xlabel, ylabel, title, title_y=1.1 if spans else 1.0)
    ax.margins(x=0.15)
    _direct_labels(ax, [(x + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y, t, c) for x, y, t, c in ends])
    _legend_below(ax, ncol=len(curves))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    out = pd.DataFrame(table)
    out.index.name = xlabel
    out.to_csv(str(path).rsplit(".", 1)[0] + ".csv")
    return out


def plot_stage_trace(yields: Sequence[float], stages: Sequence[str], path, title: str) -> None:
    """One campaign: each experiment's yield (dots) and best-so-far (line), with stage spans shaded."""
    yields = np.asarray(yields, dtype=float)
    xs = np.arange(1, len(yields) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    stages = list(stages)
    spans, start = [], 0
    for s in pd.unique(pd.Series(stages)):
        end = start + stages.count(s)
        spans.append((s, start + 0.5, end + 0.5))
        start = end
    _stage_bands(ax, spans)
    ax.scatter(xs, yields, s=14, color=SLOTS[0], alpha=0.55, linewidths=0, label="Single experiment")
    ax.plot(xs, np.maximum.accumulate(yields), color=INK, linewidth=2, label="Best so far")
    _style(ax, "Experiment number", "Yield (%)", title, title_y=1.1)
    _legend_below(ax, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    pd.DataFrame({"experiment": xs, "stage": stages, "yield": yields}).to_csv(str(path).rsplit(".", 1)[0] + ".csv",
                                                                             index=False)


def plot_grouped_means(groups: Mapping[str, Mapping[str, Sequence[float]]], path, title: str, ylabel: str,
                       xlabel: str = "", reference_lines: Mapping[str, float] | None = None) -> None:
    """Mean ± 95% CI per strategy within each group (e.g. reference-quality bins), side by side."""
    names = list(dict.fromkeys(n for g in groups.values() for n in g))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(min(13, max(7.5, 1.6 * len(groups))), 5.0))
    rows = []
    for gi, (gname, by_strat) in enumerate(groups.items()):
        for si, name in enumerate(names):
            v = np.asarray(by_strat.get(name, []), dtype=float)
            if v.size == 0:
                continue
            m, lo, hi = (x[0] for x in _mean_ci(v[:, None]))
            x = gi - 0.4 + width * (si + 0.5)
            colour = COLOURS.get(name, SLOTS[si % len(SLOTS)])
            ax.bar(x, m, width=width * 0.85, color=colour, alpha=0.85, label=LABELS.get(name, name) if gi == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, capsize=3, lw=1.2)
            ax.annotate(f"{m:.0f}", (x, hi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                        color=INK)
            rows.append({"group": gname, "strategy": name, "n": len(v), "mean": m, "ci_low": lo, "ci_high": hi})
    for text, yv in (reference_lines or {}).items():
        ax.axhline(yv, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(text, (1, yv), xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                    ha="right", fontsize=8, color=INK_2)
    n_per = {g: max(len(v) for v in s.values()) for g, s in groups.items()}
    ax.set_xticks(range(len(groups)), [f"{g}\n(n={n_per[g]})" for g in groups], fontsize=9)
    _style(ax, xlabel, ylabel, title)
    _legend_below(ax, ncol=len(names), offset=-0.30 if xlabel else -0.16)  # clear the multi-line ticks + xlabel
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(str(path).rsplit(".", 1)[0] + ".csv", index=False)


def plot_final_comparison(values: Mapping[str, Sequence[float]], path, title: str, ylabel: str,
                          reference_lines: Mapping[str, float] | None = None) -> None:
    """Per-run final values per strategy: dots (jittered) with mean ± 95% CI."""
    fig, ax = plt.subplots(figsize=(max(7.5, 1.6 * len(values)), 4.6))
    rng = np.random.default_rng(0)
    rows = []
    for i, (name, v) in enumerate(values.items()):
        v = np.asarray(v, dtype=float)
        colour = COLOURS.get(name, SLOTS[i % len(SLOTS)])
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=16, color=colour, alpha=0.45, linewidths=0)
        m, lo, hi = (x[0] for x in _mean_ci(v[:, None]))
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=INK, ms=6, capsize=4, lw=1.5)
        ax.annotate(f"{m:.1f}", (i, hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
        rows.append({"strategy": name, "n": len(v), "mean": m, "ci_low": lo, "ci_high": hi})
    for text, yv in (reference_lines or {}).items():
        ax.axhline(yv, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(text, (1, yv), xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                    ha="right", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(values)), [LABELS.get(n, n) for n in values], fontsize=9)
    _style(ax, "", ylabel, title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(str(path).rsplit(".", 1)[0] + ".csv", index=False)
