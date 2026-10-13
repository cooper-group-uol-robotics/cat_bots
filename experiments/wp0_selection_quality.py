"""WP0 alone: how good are the reference conditions it chooses? (research log §5.9, "offline selection quality")

For each distinct given reference (Shields: 1; Suzuki: 15 substrate pairs; Cernak: 2 aryl halide / amine
pairs) run the WP0 pre-screen with 10 seeds and score the chosen reference by its **reference ceiling**:
the best objective value achievable with the fixed factors at that reference, as % of the global optimum.
Compared with the ceiling of every possible reference, i.e. choosing the reference at random.

Output: results/cross_dataset/wp0_selection_quality/ (runs.csv, summary.csv)

    python experiments/wp0_selection_quality.py
"""

import dataclasses
import sys
import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overview_pipeline_benchmarks as bench  # noqa: E402
from catbots.prescreen import select_reference  # noqa: E402

OUT = bench.RESULTS / "cross_dataset" / "wp0_selection_quality"
SEEDS = 10


def one(benchmark: str, i: int, seed: int) -> dict:
    warnings.simplefilter("ignore")
    s = bench.setup(benchmark)
    spec = dataclasses.replace(s["spec"], screen=bench.SCREEN[benchmark])
    r = s["references"][i]
    pre, lab = select_reference(s["oracle"], spec, r["reference"], s["ligand_desc"], s["substrate_rows"],
                                r["reference_substrate"], substrate_desc=s["substrate_desc"], seed=seed)
    return {"benchmark": benchmark, "reference": i, "seed": seed, "n_wp0": lab.n_experiments,
            "ceiling": bench.reference_ceiling(s, pre.reference),
            **{f"chosen_{k}": str(v) for k, v in pre.chosen.items()}}


def main():
    jobs, random_ceiling = [], {}
    for b in bench.SCREEN:
        s = bench.setup(b)
        refs = s["references"]
        given = [f for f in s["spec"].fixed if f not in bench.SCREEN[b]]
        keys = pd.DataFrame([{**{k: str(r["reference"][k]) for k in given}, "sub": r["reference_substrate"]}
                             for r in refs])
        jobs += [(b, int(i), seed) for i in keys.drop_duplicates().index for seed in range(SEEDS)]
        random_ceiling[b] = pd.Series([bench.reference_ceiling(s, r["reference"]) for r in refs])
    df = pd.DataFrame(Parallel(n_jobs=-1, verbose=5)(delayed(one)(*j) for j in jobs))
    rows = []
    for b, g in df.groupby("benchmark", sort=False):
        rc = random_ceiling[b]
        rows.append({"benchmark": b, "wp0_runs": len(g), "wp0_experiments": g["n_wp0"].mean(),
                     "wp0_ceiling_mean": g["ceiling"].mean(), "wp0_ceiling_ge95": (g["ceiling"] >= 95).mean(),
                     "random_reference_ceiling_mean": rc.mean(), "random_reference_ceiling_ge95": (rc >= 95).mean()})
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "runs.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
