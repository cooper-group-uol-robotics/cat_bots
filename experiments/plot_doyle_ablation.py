"""Compare molecular representations in the overview pipeline on Doyle arylation.

Reads results/overview_pipeline_doyle/<features>/per_reference/*.json written by
overview_pipeline_doyle.py for each feature set, and writes ablation figures + tables to
results/overview_pipeline_doyle/ablation/.

    python experiments/plot_doyle_ablation.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from catbots import load_benchmark
from catbots.plotting import plot_curves, plot_final_comparison

RESULTS = Path(__file__).resolve().parents[1] / "results" / "doyle" / "overview_pipeline"
NAMES = {"kraken_rdkit": "pipeline (Kraken + RDKit)", "chemeleon": "pipeline (CheMeleon)",
         "combined": "pipeline (combined)"}


def load(features: str) -> pd.DataFrame:
    files = sorted((RESULTS / features / "per_reference").glob("*.json"))
    return pd.DataFrame([json.loads(f.read_text()) for f in files])


def main():
    oracle = load_benchmark("doyle_arylation")
    truth = oracle.data.groupby("ligand")["yield"].mean().sort_values(ascending=False)
    rank = {lig: r for r, lig in enumerate(truth.index, start=1)}
    runs = {f: load(f) for f in NAMES if (RESULTS / f / "per_reference").exists()}
    runs = {f: d for f, d in runs.items() if len(d) == 64}
    if "kraken_rdkit" not in runs:
        raise SystemExit("need the complete kraken_rdkit run (64 references) as the reference arm")
    out = RESULTS.parent / "ablation_representation"
    out.mkdir(exist_ok=True)

    base = runs["kraken_rdkit"].set_index("reference")
    rows, final, curves = [], {}, {}
    grid = sorted({int(t) for c in base["curve_pipeline"] for t in c})

    def curve_stack(series):
        return [np.array([c.get(str(t), np.nan) for t in grid], dtype=float) for c in series]

    for f, d in runs.items():
        d = d.set_index("reference").loc[base.index]
        gen, rk = d["rec_pipeline"].map(truth), d["rec_pipeline"].map(rank)
        row = {"arm": NAMES[f], "n_experiments_mean": d["n_experiments"].mean(),
               "generality_mean": gen.mean(), "generality_sd": gen.std(), "top1": (rk == 1).mean(),
               "top3": (rk <= 3).mean(), "runs_with_cluster_collapse": int((d["n_multitask_tasks"] < 5).sum())}
        if f != "kraken_rdkit":
            g0, r0 = base["rec_pipeline"].map(truth), base["rec_pipeline"].map(rank)
            row["p_generality_vs_kraken_rdkit"] = wilcoxon(gen, g0).pvalue if (gen != g0).any() else np.nan
            wins, losses = int(((rk <= 3) & (r0 > 3)).sum()), int(((rk > 3) & (r0 <= 3)).sum())
            row["top3_wins_losses_vs_kraken_rdkit"] = f"{wins}/{losses}"
            row["p_top3_vs_kraken_rdkit"] = binomtest(wins, wins + losses).pvalue if wins + losses else np.nan
        rows.append(row)
        final[NAMES[f]] = gen.to_numpy()
        curves[NAMES[f]] = curve_stack(d["curve_pipeline"])
    for s, label in [("random", "random"), ("bo", "bo")]:  # baselines from the reference arm
        gen, rk = base[f"rec_{s}"].map(truth), base[f"rec_{s}"].map(rank)
        rows.append({"arm": label, "n_experiments_mean": base["n_experiments"].mean(), "generality_mean": gen.mean(),
                     "generality_sd": gen.std(), "top1": (rk == 1).mean(), "top3": (rk <= 3).mean()})
        final[label] = gen.to_numpy()
        curves[label] = curve_stack(base[f"curve_{s}"])

    table = pd.DataFrame(rows).set_index("arm")
    table.to_csv(out / "ablation_summary.csv")
    print(table.round(3).to_string())
    refs = {"Most general ligand (Cy-BippyPhos)": truth.max(), "Mean of all 24 ligands": truth.mean()}
    plot_curves(curves, out / "ablation_generality_vs_experiments.png",
                "Doyle arylation: recommended-ligand generality by representation",
                "True mean yield over 64 pairs (%)", reference_lines=refs, x=grid)
    plot_final_comparison(final, out / "ablation_final_generality.png",
                          "Recommended ligand after the full budget, by representation",
                          "True mean yield over 64 pairs (%)", reference_lines=refs)
    top3 = table["top3"].rename("top-3 hit rate")
    top3.to_csv(out / "ablation_top3.csv")
    print(f"\nfigures and tables -> {out}")


if __name__ == "__main__":
    main()
