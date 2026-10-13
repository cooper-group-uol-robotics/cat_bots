"""Run the overview pipeline once on a bundled benchmark, with random search and plain BO at the same budget.

A single campaign for a quick look; nothing is written to results/. The full sweeps over every reference
condition are the scripts in the README's reproduction table.

    python experiments/run_pipeline.py --benchmark shields_arylation
    python experiments/run_pipeline.py --benchmark perera_suzuki --reference 5 --prescreen
    python experiments/run_pipeline.py --benchmark cernak_cn --list-references

The factor roles (screened solvent, fixed factors, metal screen) and the possible reference conditions of
each benchmark are defined in ``setup()`` of overview_pipeline_benchmarks.py / overview_pipeline_doyle.py.
Approximate time per campaign including baselines: Shields ~5 s, Doyle and Suzuki ~1-2 min,
Cernak ~35 min (mostly plain BO over 29.6k candidates; use --no-baselines to skip it).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overview_pipeline_benchmarks as bench  # noqa: E402
import overview_pipeline_doyle as doyle  # noqa: E402
from catbots.campaign import run_bo, run_random  # noqa: E402
from catbots.pipeline import recommend_general, run_ligand_substrate_pipeline, run_overview_pipeline  # noqa: E402
from catbots.prescreen import select_reference  # noqa: E402

BENCHMARKS = ("shields_arylation", "doyle_arylation", "perera_suzuki", "cernak_cn")


def _references(benchmark: str, prescreen: bool):
    if benchmark == "doyle_arylation":
        oracle, ligand_desc, substrate_desc, substrate_rows, X = doyle.setup("combined")
        return {"oracle": oracle, "ligand_desc": ligand_desc, "substrate_desc": substrate_desc,
                "substrate_rows": substrate_rows, "X": X,
                "references": [{"reference_substrate": p, "reference": {}} for p in substrate_desc.index]}
    return bench.setup(benchmark, prescreen)


def _baselines(o, X, n, seed, score):
    out = {}
    for name, runner in [("random", run_random), ("bo", run_bo)]:
        o.reset_budget()
        t0 = time.perf_counter()
        c = runner(o, X, n, seed)
        out[name] = (*score(list(c.rows), np.asarray(c.yields, dtype=float)), time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", choices=BENCHMARKS, default="shields_arylation")
    ap.add_argument("--reference", type=int, default=0, help="index of the reference condition (see --list-references)")
    ap.add_argument("--prescreen", action="store_true", help="choose the fixed factors with the WP0 pre-screen")
    ap.add_argument("--no-baselines", action="store_true", help="run the pipeline only")
    ap.add_argument("--list-references", action="store_true", help="print the reference conditions and exit")
    args = ap.parse_args()
    if args.prescreen and args.benchmark == "doyle_arylation":
        ap.error("Doyle has no fixed factors for WP0 to choose")

    s = _references(args.benchmark, args.prescreen)
    refs = s["references"]
    if args.list_references:
        for i, r in enumerate(refs):
            print(i, r.get("reference_substrate") or "", {k: str(v) for k, v in r["reference"].items()},
                  f"seed {r['seed']}" if "seed" in r else "")
        return
    if not 0 <= args.reference < len(refs):
        ap.error(f"--reference must be in 0..{len(refs) - 1}")
    r, o, X = refs[args.reference], s["oracle"], s["X"]
    print(f"[{args.benchmark}] {len(o):,} reactions; reference {args.reference} of {len(refs)}: "
          f"{r.get('reference_substrate') or ''} {({k: str(v) for k, v in r['reference'].items()})}")

    t0 = time.perf_counter()
    if args.benchmark == "doyle_arylation":
        res = run_ligand_substrate_pipeline(o, s["ligand_desc"], s["substrate_desc"], s["substrate_rows"],
                                            r["reference_substrate"], X, seed=0)
        truth = o.data.groupby("ligand")["yield"].mean()
        objective = "generality (true mean yield over all 64 substrate pairs of the recommended ligand)"
        pipeline_value, pipeline_rec = float(truth[res.recommended]), res.recommended

        def score(rows, ys):
            rec = recommend_general(o, X, rows, ys, seed=0)
            return float(truth[rec]), rec
    else:
        spec, lab, reference = s["spec"], None, r["reference"]
        if args.prescreen:
            pre, lab = select_reference(o, spec, reference, s["ligand_desc"], s["substrate_rows"],
                                        r["reference_substrate"], substrate_desc=s["substrate_desc"], seed=r["seed"])
            reference = pre.reference
            print(f"WP0 chose: {({k: str(v) for k, v in pre.chosen.items()})}")
        res = run_overview_pipeline(o, spec, reference, s["ligand_desc"], X, substrate_desc=s["substrate_desc"],
                                    substrate_rows=s["substrate_rows"], reference_substrate=r["reference_substrate"],
                                    seed=0, lab=lab)
        if spec.objective == "generality":
            truth = bench.truth_tables(s)
            cond = truth["cond"]
            objective = "generality (true mean yield over all substrate pairs of the recommended conditions)"
            pipeline_rec = bench._cond_key(res.recommended, cond)
            pipeline_value = float(truth["generality"][pipeline_rec])

            def score(rows, ys):
                rec = tuple(np.atleast_1d(recommend_general(o, X, rows, ys, factor=cond)))
                return float(truth["generality"][rec]), rec
        else:
            objective = "best yield found"
            pipeline_value, pipeline_rec = max(y for _, _, y in res.lab.log), res.recommended

            def score(rows, ys):
                return float(ys.max()), o.data.loc[rows[int(ys.argmax())], list(o.factors)].to_dict()
    seconds = time.perf_counter() - t0

    n = res.n_experiments
    print(f"\nPipeline: {n} experiments ({n / len(o):.1%} of the space), per stage {res.lab.stage_counts()}")
    print(f"Recommended: {pipeline_rec}")
    print(f"\nObjective: {objective}")
    print(f"  {'pipeline':<9}{pipeline_value:6.1f}   ({seconds:.0f} s)")
    if not args.no_baselines:
        for name, (value, _, sec) in _baselines(o, X, n, args.reference, score).items():
            print(f"  {name:<9}{value:6.1f}   ({sec:.0f} s, same {n} experiments)")


if __name__ == "__main__":
    main()
