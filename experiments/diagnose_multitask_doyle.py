"""Diagnostic: what does the WP2C multitask step decide on Doyle? (research log §5.11)

Reruns the overview pipeline (combined representation, the §5.2 settings) for all 64 reference pairs and
records which ligands the multitask ElasticNet keeps for the final space (top 20 by predicted mean over
the cluster representatives), against the ligands' true generality ranks.

Output: results/doyle/multitask_diagnostic/ (runs.csv, summary.csv)

    python experiments/diagnose_multitask_doyle.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overview_pipeline_doyle as doyle  # noqa: E402
from catbots.pipeline import run_ligand_substrate_pipeline  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "doyle" / "multitask_diagnostic"


def one(i: int) -> dict:
    warnings.simplefilter("ignore")
    o, lig, sub, rows, X = doyle.setup("combined")
    truth = o.data.groupby("ligand")["yield"].mean().sort_values(ascending=False)
    rank = {l: r for r, l in enumerate(truth.index, start=1)}
    res = run_ligand_substrate_pipeline(o, lig, sub, rows, sub.index[i], X, seed=0)
    d = res.decisions
    kept = d["final_ligands"]
    return {"reference": i, "n_ligands": len(truth), "n_final_ligands": len(kept),
            "n_multitask_ligands": len(d["multitask_ligands"]), "n_tasks": len(d["cluster_representatives"]),
            "n_WP2C-2": res.lab.stage_counts().get("WP2C-2", 0), "n_experiments": res.n_experiments,
            "best_ligand_kept": truth.index[0] in kept, "top3_kept": sum(l in kept for l in truth.index[:3]),
            "dropped_true_ranks": ";".join(str(rank[l]) for l in truth.index if l not in kept)}


def main():
    df = pd.DataFrame(Parallel(n_jobs=-1, verbose=5)(delayed(one)(i) for i in range(64)))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "runs.csv", index=False)
    ranks = pd.Series([int(x) for s in df["dropped_true_ranks"] if s for x in s.split(";")])
    summary = pd.DataFrame([{
        "runs": len(df), "ligands": df["n_ligands"].iloc[0], "final_ligands": df["n_final_ligands"].mean(),
        "multitask_ligands": df["n_multitask_ligands"].mean(), "tasks": df["n_tasks"].mean(),
        "experiments_WP2C-2": df["n_WP2C-2"].mean(), "experiments_total": df["n_experiments"].mean(),
        "best_ligand_kept": df["best_ligand_kept"].mean(), "top3_kept_of_3": df["top3_kept"].mean(),
        "best_ligand_dropped_runs": int((ranks == 1).sum()),
    }])
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.T.to_string(header=False))
    print("true ranks of dropped ligands:", ranks.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
