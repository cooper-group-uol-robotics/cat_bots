"""Generality-aware BO baseline vs the overview pipeline, on the benchmarks with a generality objective.

Plain BO (the baseline elsewhere) maximises single-reaction yield. This baseline (``run_generality_bo``)
optimises what the pipeline is judged on: the mean yield over substrates of one condition vector. It is
run at exactly each cached pipeline run's experiment count (same seed as that run's plain-BO baseline),
and its recommendation uses the same GP rule (``recommend_general``) as the other baselines.

  doyle   (64 runs; condition = ligand)        reads results/doyle/overview_pipeline/combined/
  suzuki  (480 runs; condition = ligand, base, solvent)  reads results/suzuki/overview_pipeline/combined/

Output: results/<dataset>/generality_bo_baseline/ (runs.csv, summary.csv, final_generality.png).

    python experiments/generality_bo_baseline.py --dataset {doyle,suzuki}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catbots import load_benchmark  # noqa: E402
from catbots.campaign import run_generality_bo  # noqa: E402
from catbots.featurizers import encode_conditions  # noqa: E402
from catbots.pipeline import recommend_general  # noqa: E402
from catbots.plotting import plot_final_comparison  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
BENCH = {"doyle": "doyle_arylation", "suzuki": "perera_suzuki"}
SUBSTRATES = ("electrophile", "nucleophile")
STRATEGIES = ("pipeline", "random", "bo", "gbo")


def setup(dataset: str):
    o = load_benchmark(BENCH[dataset])
    cond = [f for f in o.factors if f not in SUBSTRATES]
    X = encode_conditions(o.search_space(), o.factors)
    gen = o.data.groupby(cond)["yield"].mean().sort_values(ascending=False)
    return o, cond, X, gen


def cached_runs(dataset: str) -> list[tuple[int, dict]]:
    files = sorted((RESULTS / dataset / "overview_pipeline" / "combined" / "per_reference").glob("*.json"))
    return [(int(f.stem), json.loads(f.read_text())) for f in files]


def _key(rec):
    return tuple(rec) if isinstance(rec, (list, tuple)) else (rec,)


def lookup(gen: pd.Series, rec):
    """True generality of a recommended condition (a level, or a tuple for multi-factor conditions)."""
    k = _key(rec)
    return float(gen[k if gen.index.nlevels > 1 else k[0]])


def run_one(dataset: str, i: int, prev: dict) -> dict:
    """gBO for one pipeline run. Only gBO's own result is cached; the pipeline / random / plain-BO values are
    always re-read from the pipeline run, so a rerun pipeline is picked up without recomputing gBO."""
    cache = RESULTS / dataset / "generality_bo_baseline" / "per_reference" / f"{i:03d}.json"
    o, cond, X, gen = setup(dataset)
    n = int(prev["n_experiments"])
    own = json.loads(cache.read_text()) if cache.exists() else {}
    if own.get("n_experiments") != n:  # missing, or the pipeline run's budget changed: rerun gBO
        t0 = time.perf_counter()
        r = run_generality_bo(o, X, n, i, cond)
        rec = recommend_general(o, X, r.rows, r.yields, factor=cond if len(cond) > 1 else cond[0])
        own = {"index": i, "n_experiments": n, "seconds_gbo": time.perf_counter() - t0,
               "gen_gbo": lookup(gen, rec), "rec_gbo": list(_key(rec))}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(own))
    out = {k: own[k] for k in ("index", "n_experiments", "seconds_gbo", "gen_gbo", "rec_gbo")}
    for s in ("pipeline", "random", "bo"):  # from the pipeline run (Doyle stores the recommended ligand)
        out[f"gen_{s}"] = float(prev[f"gen_{s}"]) if f"gen_{s}" in prev else lookup(gen, prev[f"rec_{s}"])
    if "best_generality_with_reference_base" in prev:
        out["reference_ceiling"] = 100 * prev["best_generality_with_reference_base"] / gen.max()
    return out


def summarise(df: pd.DataFrame, gen: pd.Series) -> pd.DataFrame:
    rank = {k: r for r, k in enumerate(gen.index, start=1)}
    rows = {}
    for s in STRATEGIES:
        v = df[f"gen_{s}"]
        rows[s] = {"n": len(v), "generality_mean": v.mean(), "generality_sd": v.std(),
                   "top1_found": float(np.isclose(v, gen.iloc[0]).mean())}
        if s != "pipeline":
            d = df["gen_pipeline"] - v
            rows[s] |= {"pipeline_minus_this": d.mean(),
                        "p_vs_pipeline": float(wilcoxon(df["gen_pipeline"], v).pvalue) if (d != 0).any() else np.nan,
                        "pipeline_wins_losses": f"{(d > 0).sum()}/{(d < 0).sum()}"}
    if "seconds_gbo" in df:
        rows["gbo"]["seconds_mean"] = df["seconds_gbo"].mean()
    return pd.DataFrame(rows).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(BENCH), required=True)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    runs = cached_runs(args.dataset)
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_one)(args.dataset, i, prev) for i, prev in runs)
    df = pd.DataFrame(rows).sort_values("index")
    out = RESULTS / args.dataset / "generality_bo_baseline"
    df.to_csv(out / "runs.csv", index=False)
    _, _, _, gen = setup(args.dataset)
    parts = {"all references": df}
    if "reference_ceiling" in df:
        parts["reference ceiling >= 95%"] = df[df["reference_ceiling"] >= 95]
    with open(out / "summary.csv", "w", encoding="utf-8") as fh:
        for name, part in parts.items():
            s = summarise(part, gen)
            print(f"\n[{args.dataset}] {name} (n = {len(part)})\n{s.to_string()}")
            fh.write(f"# {name} (n = {len(part)})\n")
            s.to_csv(fh)
    title = {"doyle": "Doyle arylation", "suzuki": "Perera Suzuki"}[args.dataset]
    plot_final_comparison({s: df[f"gen_{s}"] for s in STRATEGIES}, out / "final_generality.png",
                          f"{title}: generality of the recommended condition, budget-matched",
                          "True mean yield over all substrates (%)",
                          reference_lines={"Most general condition": gen.iloc[0]})
    print(f"\n{len(df)} runs in {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
