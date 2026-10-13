"""First benchmark: does screen -> reduce -> BO beat BO on the full space at equal budget?

SUPERSEDED (exploratory only): budgets here are a fraction of the space or the EDBO/GOLLuM 60, not the
project overview's stage batches. The overview pipeline itself is run by overview_pipeline_*.py.

Each strategy gets the same number of oracle queries. By default the budget follows the project
overview: fewer than 800 experiments in an 88,000-reaction space, i.e. 0.9% of each benchmark's
space (16 for Shields arylation, 34 for the EDBO Suzuki subset). ``--budget 60`` reproduces the
EDBO/GOLLuM convention (10 initial + 50 BO iterations) for comparison with the literature.

    python experiments/first_benchmark.py --seeds 100                # overview budget (0.9%)
    python experiments/first_benchmark.py --seeds 100 --budget 60    # literature budget
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

from catbots import load_benchmark
from catbots.campaign import best_so_far_curves, run_bo, run_random, run_screen_bo, summarise
from catbots.featurizers import encode_conditions

BENCHMARKS = ["shields_arylation", "perera_suzuki_edbo"]
STRATEGIES = {
    "random": lambda o, X, b, s: run_random(o, X, b, s),
    "bo": lambda o, X, b, s: run_bo(o, X, b, s, n_init=N_INIT, init="random"),
    "bo_kennard_stone": lambda o, X, b, s: run_bo(o, X, b, s, n_init=N_INIT, init="kennard_stone"),
    "screen3_bo": lambda o, X, b, s: run_screen_bo(o, X, b, s, factor="ligand", keep=3),
}
OVERVIEW_FRACTION = 800 / 88_000  # project overview: <800 experiments over the 88k-reaction design space
N_INIT = 10
RESULTS = Path(__file__).resolve().parents[1] / "results" / "exploratory_screen_then_bo"


def _one(name, strategy, budget, seed):
    oracle = load_benchmark(name)
    X = encode_conditions(oracle.search_space(), oracle.factors)
    result = STRATEGIES[strategy](oracle, X, budget, seed)
    assert oracle.n_queries == budget, (name, strategy, oracle.n_queries)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=None, help="absolute budget for every benchmark")
    ap.add_argument("--budget-fraction", type=float, default=OVERVIEW_FRACTION,
                    help="budget as a fraction of each benchmark's space (ignored if --budget is given)")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    t0 = time.time()
    oracles = {b: load_benchmark(b) for b in BENCHMARKS}
    budgets = {b: args.budget or round(args.budget_fraction * len(o)) for b, o in oracles.items()}
    for b, n in budgets.items():
        n_screen = oracles[b].data["ligand"].nunique()
        if n <= max(N_INIT, n_screen):
            raise SystemExit(f"{b}: budget {n} leaves no BO iterations after {max(N_INIT, n_screen)} initial/screen runs")
        print(f"{b}: budget {n} experiments ({n / len(oracles[b]):.2%} of {len(oracles[b])})")
    jobs = [(b, s, budgets[b], seed) for b in BENCHMARKS for s in STRATEGIES for seed in range(args.seeds)]
    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(_one)(*j) for j in jobs)

    runs = pd.concat([summarise([r for r in results if r.benchmark == b], oracles[b]) for b in BENCHMARKS])
    OUT = RESULTS / (f"budget{args.budget}" if args.budget else f"fraction{args.budget_fraction:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / "runs.csv", index=False)
    best_so_far_curves(results).to_csv(OUT / "curves.csv", index=False)

    table = runs.groupby(["benchmark", "strategy"], sort=False).agg(
        best_yield_mean=("best_yield", "mean"),
        best_yield_sd=("best_yield", "std"),
        found_global_max=("found_global_max", "mean"),
        top5_coverage_mean=("top_coverage", "mean"),
        top5_coverage_sd=("top_coverage", "std"),
        kept_contains_best=("kept_levels_contain_best", "mean"),
    ).round(3)
    # Welch t-test of each strategy against plain BO on the same benchmark
    for (b, s), g in runs.groupby(["benchmark", "strategy"], sort=False):
        ref = runs[(runs.benchmark == b) & (runs.strategy == "bo")]
        for metric, col in [("top_coverage", "p_coverage_vs_bo"), ("best_yield", "p_best_vs_bo")]:
            table.loc[(b, s), col] = np.nan if s == "bo" else round(
                ttest_ind(g[metric], ref[metric], equal_var=False).pvalue, 4)
    table.to_csv(OUT / "summary.csv")
    print(table.to_string())
    print(f"\n{len(results)} campaigns in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
