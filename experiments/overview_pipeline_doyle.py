"""Overview pipeline (WP2A-C + final BO) on the Doyle arylation grid vs random and plain BO at equal budget.

Batches follow project_overview.md, capped at the pool sizes of this benchmark (24 ligands, 64 substrate
pairs, one metal). The overview fixes "substrate 1" by expert choice; here the pipeline is run once
for every possible reference pair, and each baseline gets the same number of experiments as that run.
Metric: generality of the recommended ligand = its true mean yield over all 64 substrate pairs.

``--features`` picks the molecular representation used by the reduction stages (KS selection,
models, classifier, clustering); the BO surrogate always uses one-hot conditions:
  kraken_rdkit  Kraken (reduced, 34) for ligands; RDKit 2D (|r| < 0.7) for both coupling partners
  chemeleon     CheMeleon embeddings, PCA to 90% variance per molecule set
  combined      both, concatenated

    python experiments/overview_pipeline_doyle.py --features kraken_rdkit
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker; also avoids KMeans' MKL leak on Windows

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest, wilcoxon

from catbots import load_benchmark
from catbots.campaign import run_bo, run_random
from catbots.featurizers import CheMeleonFeaturizer, KrakenLigandFeaturizer, encode_conditions
from catbots.featurizers.rdkit_desc import rdkit_descriptors
from catbots.pipeline import recommend_general, run_ligand_substrate_pipeline
from catbots.plotting import plot_curves, plot_final_comparison, plot_stage_trace
from catbots.preprocessing import drop_correlated

FEATURES = ("kraken_rdkit", "chemeleon", "combined")
STRATEGIES = ("pipeline", "random", "bo")
CHECKPOINT_STEP = 20  # generality of the recommendation is re-evaluated every 20 experiments
RESULTS = Path(__file__).resolve().parents[1] / "results" / "doyle" / "overview_pipeline"


def _pair_table(oracle):
    pairs = oracle.data[["electrophile", "nucleophile"]].drop_duplicates()
    pairs.index = (pairs["electrophile"] + "|" + pairs["nucleophile"]).to_numpy()
    return pairs


def _partner_block(pairs, table_e, table_n):
    return pd.concat([table_e.loc[pairs["electrophile"]].set_axis(pairs.index),
                      table_n.loc[pairs["nucleophile"]].set_axis(pairs.index)], axis=1)


def setup(features: str = "kraken_rdkit"):
    oracle = load_benchmark("doyle_arylation")
    ligands = oracle.levels("ligand")
    pairs = _pair_table(oracle)
    blocks_l, blocks_s = [], []
    if features in ("kraken_rdkit", "combined"):
        blocks_l.append(KrakenLigandFeaturizer("reduced").transform(ligands))
        raw = _partner_block(pairs, rdkit_descriptors(pairs["electrophile"]).add_prefix("elec_"),
                             rdkit_descriptors(pairs["nucleophile"]).add_prefix("nuc_"))
        blocks_s.append(raw[drop_correlated(raw, threshold=0.7)])
    if features in ("chemeleon", "combined"):
        cm = CheMeleonFeaturizer()
        blocks_l.append(cm.transform_pca(ligands, prefix="lig_cm_pc"))
        blocks_s.append(_partner_block(pairs, cm.transform_pca(pairs["electrophile"].unique(), prefix="elec_cm_pc"),
                                       cm.transform_pca(pairs["nucleophile"].unique(), prefix="nuc_cm_pc")))
    if not blocks_l:
        raise ValueError(f"features must be one of {FEATURES}")
    ligand_desc = pd.concat(blocks_l, axis=1)
    substrate_desc = pd.concat(blocks_s, axis=1)
    substrate_rows = pairs[["electrophile", "nucleophile"]].to_dict("index")
    X = encode_conditions(oracle.search_space(), oracle.factors)
    return oracle, ligand_desc, substrate_desc, substrate_rows, X


def _generality_curve(oracle, X, rows, yields, truth, n):
    """Generality of the ligand a strategy would recommend after each checkpoint (common GP rule)."""
    ts = sorted(set(list(range(CHECKPOINT_STEP, n + 1, CHECKPOINT_STEP)) + [n]))
    return {int(t): float(truth[recommend_general(oracle, X, list(rows[:t]), np.asarray(yields[:t]))]) for t in ts}


def run_reference(i: int, features: str):
    """Run pipeline + baselines for reference pair ``i``; cached to <out>/per_reference/{i}.json."""
    out_dir = RESULTS / features
    cache = out_dir / "per_reference" / f"{i:02d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    oracle, ligand_desc, substrate_desc, substrate_rows, X = setup(features)
    truth = oracle.data.groupby("ligand")["yield"].mean()
    ref = substrate_desc.index[i]
    res = run_ligand_substrate_pipeline(oracle, ligand_desc, substrate_desc, substrate_rows, ref, X, seed=0)
    n = res.n_experiments
    p_rows = [r for r, _, _ in res.lab.log]
    p_yields = [y for _, _, y in res.lab.log]
    out = {"reference": ref, "features": features, "n_experiments": n,
           **{f"n_{k}": v for k, v in res.lab.stage_counts().items()},
           "final_space": res.decisions["final_space_size"], "n_active": len(res.decisions["active_substrates"]),
           "n_multitask_tasks": len(res.decisions["cluster_representatives"]),
           "n_final_substrates": len(res.decisions["final_substrates"]),
           "classifier_features": ";".join(res.diagnostics["classifier_features"]),
           "wp2b1_model": res.diagnostics["wp2b1_model"],
           "best_ligand_at_reference": res.decisions["best_ligand"], "rec_pipeline": res.recommended,
           "trace_pipeline": p_yields, "stages_pipeline": [s for _, s, _ in res.lab.log],
           "curve_pipeline": _generality_curve(oracle, X, p_rows, p_yields, truth, n)}
    for name, runner in [("random", run_random), ("bo", run_bo)]:
        oracle.reset_budget()
        r = runner(oracle, X, n, i)
        out[f"rec_{name}"] = recommend_general(oracle, X, r.rows, r.yields, seed=0)
        out[f"trace_{name}"] = [float(y) for y in r.yields]
        out[f"curve_{name}"] = _generality_curve(oracle, X, r.rows, r.yields, truth, n)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, default=str))
    return out


def summarise(df: pd.DataFrame, oracle) -> pd.DataFrame:
    truth = oracle.data.groupby("ligand")["yield"].mean().sort_values(ascending=False)
    rank = {lig: r for r, lig in enumerate(truth.index, start=1)}
    for s in STRATEGIES:
        df[f"gen_{s}"] = df[f"rec_{s}"].map(truth)
        df[f"rank_{s}"] = df[f"rec_{s}"].map(rank)
    summary = pd.DataFrame({s: {
        "generality_mean": df[f"gen_{s}"].mean(), "generality_sd": df[f"gen_{s}"].std(),
        "rank_median": df[f"rank_{s}"].median(), "top1": (df[f"rank_{s}"] == 1).mean(),
        "top3": (df[f"rank_{s}"] <= 3).mean()} for s in STRATEGIES}).T
    for s in ("random", "bo"):
        summary.loc[s, "p_generality_vs_pipeline"] = wilcoxon(df["gen_pipeline"], df[f"gen_{s}"]).pvalue
        wins = int(((df["rank_pipeline"] <= 3) & (df[f"rank_{s}"] > 3)).sum())
        losses = int(((df["rank_pipeline"] > 3) & (df[f"rank_{s}"] <= 3)).sum())
        summary.loc[s, "top3_wins_losses"] = f"{wins}/{losses}"
        summary.loc[s, "p_top3_vs_pipeline"] = binomtest(wins, wins + losses).pvalue if wins + losses else np.nan
    return summary


def make_figures(df: pd.DataFrame, oracle, out_dir: Path, features: str):
    truth = oracle.data.groupby("ligand")["yield"].mean()
    refs = {"Most general ligand (Cy-BippyPhos)": truth.max(), "Mean of all 24 ligands": truth.mean()}
    tag = {"kraken_rdkit": "Kraken + RDKit", "chemeleon": "CheMeleon", "combined": "Kraken + RDKit + CheMeleon"}[features]
    stage_ends = df[[c for c in ["n_WP2A", "n_WP2B-1", "n_WP2B-2", "n_WP2C-1", "n_WP2C-2", "n_BO"] if c in df]]
    bounds = stage_ends.fillna(0).median().cumsum()
    bounds = {k[2:]: v for k, v in bounds.items() if stage_ends[k].median() > 0}

    plot_curves({s: [np.maximum.accumulate(np.asarray(t)) for t in df[f"trace_{s}"]] for s in STRATEGIES},
                out_dir / "best_so_far_yield.png", f"Doyle arylation: best yield found ({tag})",
                "Best yield so far (%)", stage_bounds=bounds)
    grid = sorted({int(t) for c in df["curve_pipeline"] for t in c})
    curves = {s: [np.array([c.get(str(t), c.get(t, np.nan)) for t in grid], dtype=float) for c in df[f"curve_{s}"]]
              for s in STRATEGIES}
    plot_curves(curves, out_dir / "generality_vs_experiments.png",
                f"Doyle arylation: generality of the recommended ligand ({tag})",
                "True mean yield over 64 pairs (%)", reference_lines=refs, stage_bounds=bounds, x=grid)
    plot_final_comparison({s: df[f"gen_{s}"] for s in STRATEGIES}, out_dir / "final_generality.png",
                          f"Recommended ligand after the full budget ({tag})", "True mean yield over 64 pairs (%)",
                          reference_lines=refs)
    typical = (df["n_experiments"] - df["n_experiments"].median()).abs().idxmin()
    plot_stage_trace(df.loc[typical, "trace_pipeline"], df.loc[typical, "stages_pipeline"],
                     out_dir / "pipeline_trace_example.png",
                     f"One pipeline run by stage (reference {typical}, {tag})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=FEATURES, default="kraken_rdkit")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--references", type=int, default=None, help="limit to the first N reference pairs")
    args = ap.parse_args()
    t0 = time.time()
    oracle, ligand_desc, substrate_desc, _, _ = setup(args.features)
    print(f"[{args.features}] ligand descriptors {ligand_desc.shape}, substrate descriptors {substrate_desc.shape}")
    refs = range(args.references or len(substrate_desc))
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_reference)(i, args.features) for i in refs)
    df = pd.DataFrame(rows)
    out_dir = RESULTS / args.features
    summary = summarise(df, oracle)
    df.drop(columns=[c for c in df if c.startswith(("trace_", "stages_", "curve_"))]).to_csv(out_dir / "runs.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")
    make_figures(df, oracle, out_dir, args.features)
    stage_cols = [c for c in df.columns if c.startswith("n_WP") or c == "n_BO"]
    print(f"Experiments per run: mean {df['n_experiments'].mean():.1f} "
          f"(min {df['n_experiments'].min()}, max {df['n_experiments'].max()}) | per stage (mean): "
          f"{df[stage_cols].mean().round(1).to_dict()} | multitask tasks: {df['n_multitask_tasks'].value_counts().to_dict()}")
    print(summary.round(3).to_string())
    print(f"\n{len(df)} reference pairs in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
