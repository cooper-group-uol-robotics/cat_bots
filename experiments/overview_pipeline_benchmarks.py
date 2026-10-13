"""Overview pipeline vs random and plain BO on the other benchmarks, at equal budget.

Same rules as overview_pipeline_doyle.py: the overview's batches capped at pool size; BO = 25% of the
reduced space; factors outside the overview's design (base; Shields also concentration/temperature;
Cernak base/temperature/aryl halide) are fixed at the run's reference level; solvent is screened at
the reference level and opened in the final space. Molecules use the "combined" representation
(CheMeleon PCA + Kraken/RDKit), the most robust arm of the Doyle ablation. Every possible reference
condition is run once; each baseline gets the same number of experiments as that pipeline run.

Objectives
  perera_suzuki  (15 substrate pairs)  generality: true mean yield over all 15 pairs of the recommended
                                       condition vector (ligand, base, solvent)
  shields_arylation, cernak_cn         single substrate: best yield found

    python experiments/overview_pipeline_benchmarks.py --benchmark perera_suzuki
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest, wilcoxon

from catbots import NONE, load_benchmark
from catbots.campaign import run_bo, run_random
from catbots.featurizers import combined_ligand_descriptors, combined_molecule_descriptors, encode_conditions
from catbots.pipeline import PipelineSpec, recommend_general, run_overview_pipeline
from catbots.plotting import plot_curves, plot_final_comparison, plot_stage_trace
from catbots.prescreen import select_reference

STRATEGIES = ("pipeline", "random", "bo")
# --prescreen: factors WP0 chooses (the rest of spec.fixed, e.g. the aryl halide, stays given) and seeds per given reference
SCREEN = {"shields_arylation": ("base", "concentration", "temperature"), "perera_suzuki": ("base",),
          "cernak_cn": ("base", "temperature")}
PRESCREEN_SEEDS = {"shields_arylation": 50, "perera_suzuki": 8, "cernak_cn": 20}
CHECKPOINT_STEP = 20
RESULTS = Path(__file__).resolve().parents[1] / "results"
SHORT = {"perera_suzuki": "suzuki", "shields_arylation": "shields", "cernak_cn": "cernak"}  # results/<dataset>/


# ----------------------------------------------------------------------------------------------- setups
def setup(benchmark: str, prescreen: bool = False) -> dict:
    o = load_benchmark(benchmark)
    ligands = [l for l in o.levels("ligand") if l != NONE]
    ligand_desc = combined_ligand_descriptors(ligands)
    X = encode_conditions(o.search_space(), o.factors)
    s = {"oracle": o, "ligand_desc": ligand_desc, "X": X, "substrate_desc": None, "substrate_rows": None}
    if benchmark == "perera_suzuki":
        spec = PipelineSpec(substrates=("electrophile", "nucleophile"), solvent="solvent", fixed=("base",))
        pairs = o.data[["electrophile", "nucleophile"]].drop_duplicates()
        pairs.index = (pairs["electrophile"] + "|" + pairs["nucleophile"]).to_numpy()
        E = combined_molecule_descriptors(pairs["electrophile"], "elec")
        N = combined_molecule_descriptors(pairs["nucleophile"], "nuc")
        s["substrate_desc"] = pd.concat([E.loc[pairs["electrophile"]].set_axis(pairs.index),
                                         N.loc[pairs["nucleophile"]].set_axis(pairs.index)], axis=1)
        s["substrate_rows"] = pairs[["electrophile", "nucleophile"]].to_dict("index")
        refs = [{"reference_substrate": p, "reference": {"base": bse, "solvent": sv}}
                for p in pairs.index for bse in o.levels("base") for sv in o.levels("solvent")]
    elif benchmark == "shields_arylation":
        spec = PipelineSpec(solvent="solvent", fixed=("base", "concentration", "temperature"))
        refs = [{"reference_substrate": None, "reference": dict(zip(["base", "concentration", "temperature", "solvent"], k))}
                for k in o.data[["base", "concentration", "temperature", "solvent"]].drop_duplicates().itertuples(index=False)]
    elif benchmark == "cernak_cn":
        spec = PipelineSpec(metal="metal_complex", metal_group="metal", solvent="solvent",
                            fixed=("base", "temperature", "electrophile", "nucleophile"))
        key = ["solvent", "base", "temperature", "electrophile", "nucleophile"]
        has_ctrl = o.data[o.data["metal_complex"] == NONE][key].drop_duplicates()
        refs = [{"reference_substrate": None, "reference": dict(zip(key, k))} for k in has_ctrl.itertuples(index=False)]
    else:
        raise ValueError(benchmark)
    if prescreen:  # WP0 chooses the screened factors and the solvent: one reference per given level x seed
        spec = dataclasses.replace(spec, screen=SCREEN[benchmark])
        given = [f for f in spec.fixed if f not in spec.screen]
        seen, base = set(), []
        for r in refs:
            key = (r["reference_substrate"], *(str(r["reference"][f]) for f in given))
            if key not in seen:
                seen.add(key)
                base.append(r)
        refs = [{**r, "seed": sd} for r in base for sd in range(PRESCREEN_SEEDS[benchmark])]
    s["spec"], s["references"] = spec, refs
    return s


def truth_tables(s: dict) -> dict:
    o, spec = s["oracle"], s["spec"]
    y = o.data[o.target]
    if spec.objective == "generality":
        cond = [f for f in o.factors if f not in spec.substrates]
        gen = o.data.groupby(cond)["yield"].mean().sort_values(ascending=False)
        return {"cond": cond, "generality": gen, "rank": {k: r for r, k in enumerate(gen.index, start=1)}}
    return {"global_max": float(y.max()), "top1pct": set(o.data.index[y >= y.quantile(0.99)])}


# ---------------------------------------------------------------------------------------------- one run
def _cond_key(rec: dict, cond: list[str]) -> tuple:
    return tuple(rec[c] for c in cond)


def reference_ceiling(s: dict, reference: dict) -> float:
    """Best objective value that exists with the fixed factors at ``reference``, as % of the global optimum."""
    o, spec = s["oracle"], s["spec"]
    if spec.objective == "generality":
        gen = truth_tables(s)["generality"]
        sel = gen
        for f in spec.fixed:
            sel = sel[sel.index.get_level_values(f) == reference[f]]
        return float(100 * sel.max() / gen.max())
    mask = pd.Series(True, index=o.data.index)
    for f in spec.fixed:
        mask &= o.data[f] == reference[f]
    return float(100 * o.data.loc[mask, o.target].max() / o.data[o.target].max())


def variant(prescreen: bool) -> str:
    return "overview_pipeline_wp0" if prescreen else "overview_pipeline"


def run_reference(benchmark: str, i: int, prescreen: bool = False) -> dict:
    out_dir = RESULTS / SHORT[benchmark] / variant(prescreen) / "combined" / "per_reference"
    cache = out_dir / f"{i:03d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    s = setup(benchmark, prescreen)
    o, spec, X, truth = s["oracle"], s["spec"], s["X"], truth_tables(s)
    r = s["references"][i]
    t0 = time.perf_counter()
    lab, reference, wp0 = None, r["reference"], {}
    if prescreen:
        pre, lab = select_reference(o, spec, r["reference"], s["ligand_desc"], s["substrate_rows"],
                                    r["reference_substrate"], substrate_desc=s["substrate_desc"], seed=r["seed"])
        reference = pre.reference
        wp0 = {"seed": r["seed"], "wp0_chosen": {k: str(v) for k, v in pre.chosen.items()},
               "wp0_candidates": pre.n_candidates, "wp0_probes": len(pre.probes)}
    res = run_overview_pipeline(o, spec, reference, s["ligand_desc"], X, substrate_desc=s["substrate_desc"],
                                substrate_rows=s["substrate_rows"], reference_substrate=r["reference_substrate"],
                                seed=0, lab=lab)
    seconds = {"pipeline": time.perf_counter() - t0}
    n = res.n_experiments
    p_rows, p_yields = [x for x, _, _ in res.lab.log], [y for _, _, y in res.lab.log]
    out = {"index": i, "reference": {k: str(v) for k, v in reference.items()}, **wp0,
           "reference_ceiling": reference_ceiling(s, reference),
           "reference_substrate": r["reference_substrate"], "n_experiments": n,
           **{f"n_{k}": v for k, v in res.lab.stage_counts().items()},
           "final_space": res.decisions["final_space_size"],
           "metals_kept": res.decisions.get("metals_kept"),
           "trace_pipeline": p_yields, "stages_pipeline": [st for _, st, _ in res.lab.log]}
    traces = {"pipeline": (p_rows, p_yields)}
    for name, runner in [("random", run_random), ("bo", run_bo)]:
        o.reset_budget()
        t0 = time.perf_counter()
        c = runner(o, X, n, i)
        seconds[name] = time.perf_counter() - t0
        traces[name] = (list(c.rows), [float(v) for v in c.yields])
        out[f"trace_{name}"] = traces[name][1]
    if spec.objective == "generality":
        cond = truth["cond"]
        recs = {"pipeline": _cond_key(res.recommended, cond)}
        for name in ("random", "bo"):
            rows, ys = traces[name]
            recs[name] = recommend_general(o, X, rows, np.asarray(ys), factor=cond)
        for name, key in recs.items():
            key = tuple(key) if isinstance(key, (tuple, list)) else (key,)
            out[f"gen_{name}"] = float(truth["generality"][key])
            out[f"rank_{name}"] = int(truth["rank"][key])
        ts = sorted(set(list(range(CHECKPOINT_STEP, n + 1, CHECKPOINT_STEP)) + [n]))
        for name, (rows, ys) in traces.items():
            out[f"curve_{name}"] = {int(t): float(truth["generality"][tuple(np.atleast_1d(
                recommend_general(o, X, rows[:t], np.asarray(ys[:t]), factor=cond)))]) for t in ts}
        best_fixed = truth["generality"].xs(reference["base"], level="base").max()
        out["best_generality_with_reference_base"] = float(best_fixed)
    else:
        for name, (rows, ys) in traces.items():
            out[f"best_{name}"] = float(max(ys))
            out[f"top1pct_found_{name}"] = len(truth["top1pct"].intersection(rows))
        # best yield that exists inside the pipeline's reduced space (what fixing factors can cost)
        out["final_space_max"] = float(o.data.loc[res.decisions["final_rows"], o.target].max())
    out.update({f"seconds_{k}": v for k, v in seconds.items()})  # wall time (parallel workers: relative only)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, default=str))
    return out


# ------------------------------------------------------------------------------------------- summarise
def paired(a: pd.Series, b: pd.Series) -> float:
    return float(wilcoxon(a, b).pvalue) if (a != b).any() else float("nan")


def summarise(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    rows = {}
    if s["spec"].objective == "generality":
        n_cond = len(truth_tables(s)["generality"])
        top = max(1, round(0.05 * n_cond))
        for st in STRATEGIES:
            rows[st] = {"generality_mean": df[f"gen_{st}"].mean(), "generality_sd": df[f"gen_{st}"].std(),
                        "rank_median": df[f"rank_{st}"].median(), "top1": (df[f"rank_{st}"] == 1).mean(),
                        f"top5pct_(rank<={top}_of_{n_cond})": (df[f"rank_{st}"] <= top).mean()}
        for st in ("random", "bo"):
            rows[st]["p_generality_vs_pipeline"] = paired(df["gen_pipeline"], df[f"gen_{st}"])
            w = int(((df["rank_pipeline"] <= top) & (df[f"rank_{st}"] > top)).sum())
            l = int(((df["rank_pipeline"] > top) & (df[f"rank_{st}"] <= top)).sum())
            rows[st]["top5pct_wins_losses"] = f"{w}/{l}"
            rows[st]["p_top5pct_vs_pipeline"] = binomtest(w, w + l).pvalue if w + l else np.nan
    else:
        gmax = truth_tables(s)["global_max"]
        for st in STRATEGIES:
            rows[st] = {"best_yield_mean": df[f"best_{st}"].mean(), "best_yield_sd": df[f"best_{st}"].std(),
                        "found_global_max": np.isclose(df[f"best_{st}"], gmax).mean(),
                        "best_>=50%": (df[f"best_{st}"] >= 50).mean(),
                        "top1pct_found_mean": df[f"top1pct_found_{st}"].mean()}
        for st in ("random", "bo"):
            rows[st]["p_best_vs_pipeline"] = paired(df["best_pipeline"], df[f"best_{st}"])
            w = int((df["best_pipeline"] > df[f"best_{st}"]).sum())
            l = int((df["best_pipeline"] < df[f"best_{st}"]).sum())
            rows[st]["pipeline_wins_losses"] = f"{w}/{l}"
    for st in STRATEGIES:  # compute cost (runs execute in parallel, so compare strategies, not absolute values)
        if f"seconds_{st}" in df:
            rows[st]["seconds_mean"] = df[f"seconds_{st}"].mean()
    return pd.DataFrame(rows).T


def make_figures(df: pd.DataFrame, s: dict, out_dir: Path, benchmark: str):
    spec = s["spec"]
    ends = df[[c for c in ["n_WP0", "n_WP2A", "n_WP2B-1", "n_WP2B-2", "n_WP2C-1", "n_WP2C-2", "n_BO"] if c in df]]
    bounds = {k[2:]: v for k, v in ends.fillna(0).median().cumsum().items() if ends[k].median() > 0}
    title = benchmark.replace("_", " ")
    plot_curves({st: [np.maximum.accumulate(np.asarray(t)) for t in df[f"trace_{st}"]] for st in STRATEGIES},
                out_dir / "best_so_far_yield.png", f"{title}: best yield found", "Best yield so far (%)",
                stage_bounds=bounds)
    typical = (df["n_experiments"] - df["n_experiments"].median()).abs().idxmin()
    plot_stage_trace(df.loc[typical, "trace_pipeline"], df.loc[typical, "stages_pipeline"],
                     out_dir / "pipeline_trace_example.png", f"{title}: one pipeline run by stage")
    if spec.objective == "generality":
        gen = truth_tables(s)["generality"]
        refs = {"Most general condition": gen.max(), "Mean condition": gen.mean()}
        grid = sorted({int(t) for c in df["curve_pipeline"] for t in c})
        # checkpoint keys are ints in fresh runs and strings once reloaded from the JSON cache
        curves = {st: [np.array([c.get(t, c.get(str(t), np.nan)) for t in grid], dtype=float)
                       for c in df[f"curve_{st}"]] for st in STRATEGIES}
        plot_curves(curves, out_dir / "generality_vs_experiments.png",
                    f"{title}: generality of the recommended condition", "True mean yield over all substrates (%)",
                    reference_lines=refs, stage_bounds=bounds, x=grid)
        plot_final_comparison({st: df[f"gen_{st}"] for st in STRATEGIES}, out_dir / "final_generality.png",
                              f"{title}: recommended condition after the full budget",
                              "True mean yield over all substrates (%)", reference_lines=refs)
    else:
        gmax = truth_tables(s)["global_max"]
        plot_final_comparison({st: df[f"best_{st}"] for st in STRATEGIES}, out_dir / "final_best_yield.png",
                              f"{title}: best yield found with the full budget", "Best yield found (%)",
                              reference_lines={"Global maximum": gmax})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["perera_suzuki", "shields_arylation", "cernak_cn"])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--references", type=int, default=None)
    ap.add_argument("--prescreen", action="store_true", help="choose the reference with the WP0 pre-screen")
    args = ap.parse_args()
    t0 = time.time()
    s = setup(args.benchmark, args.prescreen)
    refs = range(args.references or len(s["references"]))
    print(f"[{args.benchmark}] {len(s['references'])} reference conditions; ligand descriptors "
          f"{s['ligand_desc'].shape} (dropped {s['ligand_desc'].attrs.get('dropped')})")
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_reference)(args.benchmark, i, args.prescreen) for i in refs)
    df = pd.DataFrame(rows)
    out_dir = RESULTS / SHORT[args.benchmark] / variant(args.prescreen) / "combined"
    summary = summarise(df, s)
    df.drop(columns=[c for c in df if c.startswith(("trace_", "stages_", "curve_"))]).to_csv(out_dir / "runs.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")
    make_figures(df, s, out_dir, args.benchmark)
    stage_cols = [c for c in df.columns if c.startswith("n_WP") or c == "n_BO"]
    print(f"Experiments per run: mean {df['n_experiments'].mean():.1f} (min {df['n_experiments'].min()}, "
          f"max {df['n_experiments'].max()}) = {df['n_experiments'].mean() / len(s['oracle']):.1%} of the space | "
          f"per stage: {df[stage_cols].mean().round(1).to_dict()}")
    print(summary.round(3).to_string())
    if "reference_ceiling" in df:
        c = df["reference_ceiling"]
        print(f"reference ceiling: mean {c.mean():.1f}% of optimum, >= 95% in {(c >= 95).mean():.0%} of runs")
    print(f"\n{len(df)} references in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
