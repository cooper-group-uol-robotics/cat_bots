"""Cross-dataset summary of the overview pipeline vs random and plain BO (research log §5.8).

For every run, the *reference ceiling* is the best objective value that exists once the factors fixed at the
reference are set (best yield for Shields/Cernak, best condition-vector generality for Suzuki), as a
percentage of the dataset's global optimum. Doyle fixes nothing, so its ceiling is always 100%.
The pipeline's advantage is (pipeline - plain BO) / global optimum, in percentage points.

Writes results/cross_dataset/ (summary table, small-multiple figure) and, per dataset, the
"by reference quality" bar charts into results/<dataset>/overview_pipeline/combined/.

    python experiments/cross_dataset_summary.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from catbots import load_benchmark
from catbots.plotting import GRID, INK, INK_2, SLOTS, SURFACE, _style, plot_grouped_means

R = Path(__file__).resolve().parents[1] / "results"
GOOD = 95.0  # ceiling (% of global optimum) above which a reference counts as "good"


def _ref(df):
    return pd.DataFrame([json.loads(str(r).replace("'", '"')) for r in df["reference"]])


def load_doyle():
    d = pd.read_csv(R / "doyle" / "overview_pipeline" / "combined" / "runs.csv")
    truth = load_benchmark("doyle_arylation").data.groupby("ligand")["yield"].mean()
    out = pd.DataFrame({s: d[f"rec_{s}"].map(truth) for s in ("pipeline", "random", "bo")})
    return out.assign(ceiling=100.0, n_experiments=d["n_experiments"]), float(truth.max()), "generality", len(truth)


def load_suzuki():
    d = pd.read_csv(R / "suzuki" / "overview_pipeline" / "combined" / "runs.csv")
    gen = load_benchmark("perera_suzuki").data.groupby(["ligand", "base", "solvent"])["yield"].mean()
    out = pd.DataFrame({s: d[f"gen_{s}"] for s in ("pipeline", "random", "bo")})
    return out.assign(ceiling=100 * d["best_generality_with_reference_base"] / gen.max(),
                      n_experiments=d["n_experiments"], base=_ref(d)["base"]), float(gen.max()), "generality", len(gen)


def load_shields():
    d = pd.read_csv(R / "shields" / "overview_pipeline" / "combined" / "runs.csv")
    gmax = float(load_benchmark("shields_arylation").data["yield"].max())
    out = pd.DataFrame({s: d[f"best_{s}"] for s in ("pipeline", "random", "bo")})
    # the final space spans every ligand and solvent at the fixed base/conc/T, so its max is the ceiling
    return out.assign(ceiling=100 * d["final_space_max"] / gmax, n_experiments=d["n_experiments"]), gmax, "best yield", None


def load_cernak():
    d = pd.read_csv(R / "cernak" / "overview_pipeline" / "combined" / "runs.csv")
    o = load_benchmark("cernak_cn")
    fixed = ["base", "temperature", "electrophile", "nucleophile"]  # solvent is opened in the final space
    cap = o.data.assign(**{k: o.data[k].astype(str) for k in fixed}).groupby(fixed)["yield"].max()
    ref = _ref(d)
    gmax = float(o.data["yield"].max())
    out = pd.DataFrame({s: d[f"best_{s}"] for s in ("pipeline", "random", "bo")})
    ceiling = [cap[tuple(str(r[k]) for k in fixed)] for _, r in ref.iterrows()]
    return out.assign(ceiling=100 * np.array(ceiling) / gmax, n_experiments=d["n_experiments"]), gmax, "best yield", None


DATASETS = {"Doyle C-H arylation": load_doyle, "Perera Suzuki": load_suzuki,
            "Shields arylation": load_shields, "Cernak C-N": load_cernak}
SPACE = {"Doyle C-H arylation": 1536, "Perera Suzuki": 5760, "Shields arylation": 1728, "Cernak C-N": 29572}


def compare(d: pd.DataFrame, other: str) -> dict:
    diff = d["pipeline"] - d[other]
    p = wilcoxon(d["pipeline"], d[other]).pvalue if (diff != 0).sum() > 5 else np.nan
    w, l = int((diff > 0).sum()), int((diff < 0).sum())
    return {f"vs_{other}_diff": diff.mean(), f"vs_{other}_p": p, f"vs_{other}_wins_losses": f"{w}/{l}"}


def main():
    out_dir = R / "cross_dataset"
    out_dir.mkdir(exist_ok=True)
    rows, frames = [], {}
    for name, loader in DATASETS.items():
        d, gopt, metric, _ = loader()
        frames[name] = (d, gopt)
        for subset, m in [("all references", slice(None)), (f"ceiling >= {GOOD:.0f}%", d["ceiling"] >= GOOD)]:
            s = d.loc[m]
            if len(s) < 3:
                rows.append({"dataset": name, "subset": subset, "n_references": len(s)})
                continue
            rows.append({"dataset": name, "subset": subset, "metric": metric, "n_references": len(s),
                         "experiments_mean": s["n_experiments"].mean(),
                         "coverage_%": 100 * s["n_experiments"].mean() / SPACE[name], "global_optimum": gopt,
                         "pipeline": s["pipeline"].mean(), "random": s["random"].mean(), "bo": s["bo"].mean(),
                         **compare(s, "bo"), **compare(s, "random")})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "summary_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(3).to_string(index=False))

    # small multiples: advantage over plain BO vs reference ceiling
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, (name, (d, gopt)) in zip(axes.ravel(), frames.items()):
        adv = 100 * (d["pipeline"] - d["bo"]) / gopt
        x = d["ceiling"] + np.random.default_rng(0).uniform(-0.4, 0.4, len(d))
        ax.scatter(x, adv, s=12, color=SLOTS[0], alpha=0.35, linewidths=0)
        ax.axhline(0, color=INK_2, linewidth=1)
        bins = pd.cut(d["ceiling"], [0, 50, 80, 90, 95, 100.01])
        means = adv.groupby(bins, observed=True).mean()
        centres = d["ceiling"].groupby(bins, observed=True).mean()
        ax.plot(centres, means, color=INK, linewidth=2, marker="o", ms=5, label="Bin mean")
        _style(ax, "", "", f"{name}  (n = {len(d)})")
        ax.title.set_fontsize(10)
    for ax in axes[1]:
        ax.set_xlabel("Reference ceiling (% of global optimum)", color=INK_2, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Pipeline - plain BO (% of optimum)", color=INK_2, fontsize=10)
    fig.suptitle("Where the overview pipeline beats plain BO: it depends on the fixed reference",
                 x=0.02, ha="left", color=INK, fontsize=12)
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out_dir / "advantage_vs_reference_ceiling.png", dpi=200)
    plt.close(fig)

    # per-dataset "by reference quality" bar charts (previously ad-hoc)
    for name, folder, ylabel in [("Shields arylation", "shields", "Best yield found (%)"),
                                 ("Cernak C-N", "cernak", "Best yield found (%)")]:
        d, gopt = frames[name]
        bins = pd.cut(d["ceiling"], [0, 20, 50, 80, 95, 100.01],
                      labels=["< 20%", "20-50%", "50-80%", "80-95%", ">= 95%"], right=False)
        groups = {f"ceiling {b}": {s: d.loc[bins == b, s] for s in ("pipeline", "random", "bo")}
                  for b in bins.cat.categories if (bins == b).sum() > 0}
        plot_grouped_means(groups, R / folder / "overview_pipeline" / "combined" / "by_reference_ceiling.png",
                           f"{name}: best yield found, by reference ceiling", ylabel,
                           xlabel="Best yield that exists at the fixed reference factors (% of global maximum)",
                           reference_lines={f"Global maximum ({gopt:.0f}%)": gopt})
    d, gopt = frames["Perera Suzuki"]
    short = {"[Na+].[OH-]": "NaOH", "O=C([O-])O.[Na+]": "NaHCO3", "[Cs+].[F-]": "CsF",
             "O=P([O-])([O-])[O-].[K+].[K+].[K+]": "K3PO4", "[K+].[OH-]": "KOH", "CC(C)(C)[O-].[Li+]": "LiOtBu",
             "CCN(CC)CC": "Et3N", "<none>": "no base"}
    name_of = lambda s: next((v for k, v in short.items() if sorted(k.split(".")) == sorted(str(s).split("."))), s)
    d = d.assign(base_name=d["base"].map(name_of))
    order = d.groupby("base_name")["ceiling"].first().sort_values(ascending=False)
    groups = {f"{b}\nceiling {c * gopt / 100:.0f}%": {s: d.loc[d.base_name == b, s] for s in ("pipeline", "random", "bo")}
              for b, c in order.items()}
    plot_grouped_means(groups, R / "suzuki" / "overview_pipeline" / "combined" / "by_reference_base.png",
                       "Perera Suzuki: generality of the recommended condition, by the base fixed at the reference",
                       "True mean yield over 15 pairs (%)",
                       xlabel="Reference base (ceiling = best condition vector with that base)",
                       reference_lines={f"Most general condition ({gopt:.1f}%)": gopt})
    print(f"\nfigures -> {out_dir} and the per-dataset folders")


if __name__ == "__main__":
    main()
