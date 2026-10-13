import numpy as np
import pandas as pd
import pytest

from catbots import load_benchmark
from catbots.featurizers import KrakenLigandFeaturizer, encode_conditions
from catbots.featurizers.rdkit_desc import rdkit_descriptors
from catbots.models import fit_activity_classifier, fit_multitask, select_regressor
from catbots.paths import BENCHMARK_DIR
from catbots.pipeline import Lab, OverviewBatches, PipelineSpec, run_ligand_substrate_pipeline, run_overview_pipeline
from catbots.preprocessing import drop_correlated


def test_drop_correlated_keeps_higher_variance():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    X = pd.DataFrame({"a": a, "a_copy": a * 0.5 + 1e-3 * rng.normal(size=50), "b": rng.normal(size=50),
                      "const": np.ones(50)})
    kept = drop_correlated(X, threshold=0.7)
    assert "b" in kept and "const" not in kept and len({"a", "a_copy"} & set(kept)) == 1


def test_models_on_toy_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(24, 4)), columns=list("wxyz"))
    y = 10 * X["w"].to_numpy() + rng.normal(size=24)
    choice = select_regressor(X, y)
    assert choice.feature_ranking.index[0] == "w" and set(choice.loo_r2) == {"ElasticNet", "MLP", "GPR"}
    clf = fit_activity_classifier(X, (X["x"] > 0).to_numpy())
    assert clf.features[0] == "x" and clf.predict(X).shape == (24,)
    assert fit_activity_classifier(X, np.ones(24, int)).predict(X).tolist() == [1] * 24  # single class
    Y = pd.DataFrame({"t1": X["y"] * 3, "t2": X["y"] * -2 + X["z"]})
    mt = fit_multitask(X, Y)
    assert mt.feature_ranking.index[0] == "y" and mt.predict(X).shape == (24, 2)


needs_data = pytest.mark.skipif(not BENCHMARK_DIR.exists(), reason="benchmark data not available")


@needs_data
def test_lab_never_repeats_and_get_is_strict():
    o = load_benchmark("shields_arylation")
    lab = Lab(o)
    lab.run([0, 1, 1, 2], "A")
    lab.run([2, 3], "B")
    assert lab.n_experiments == 4 and o.n_queries == 4 and lab.stage_counts() == {"A": 3, "B": 1}
    with pytest.raises(KeyError):
        lab.get([99])


@needs_data
def test_pipeline_stage_accounting_small_batches():
    o = load_benchmark("doyle_arylation")
    lig = KrakenLigandFeaturizer().transform(o.levels("ligand"))
    pairs = o.data[["electrophile", "nucleophile"]].drop_duplicates()
    pairs.index = (pairs["electrophile"] + "|" + pairs["nucleophile"]).to_numpy()
    E = rdkit_descriptors(pairs["electrophile"]).add_prefix("e_").loc[pairs["electrophile"]].set_axis(pairs.index)
    N = rdkit_descriptors(pairs["nucleophile"]).add_prefix("n_").loc[pairs["nucleophile"]].set_axis(pairs.index)
    raw = pd.concat([E, N], axis=1)
    sub = raw[drop_correlated(raw)]
    X = encode_conditions(o.search_space(), o.factors)
    small = OverviewBatches(ligand_pool=10, first_ligands=4, substrate_screen=8, substrate_clusters=3,
                            multitask_ligands=5, final_substrates=3, final_ligands=4)
    res = run_ligand_substrate_pipeline(o, lig, sub, pairs[["electrophile", "nucleophile"]].to_dict("index"),
                                        sub.index[0], X, batches=small)
    counts = res.lab.stage_counts()
    assert counts["WP2A"] == 4 and counts["WP2B-1"] == 6 and counts["WP2B-2"] <= 12 and counts["WP2C-1"] == 8
    assert counts["WP2C-2"] <= 2 * 5 and counts["BO"] == round(0.25 * 3 * 4)
    assert res.n_experiments == o.n_queries == len(set(r for r, _, _ in res.lab.log))
    assert res.decisions["final_space_size"] == 12 and res.recommended in res.decisions["final_ligands"]
    # ligands measured on every representative are ranked by their observed mean, not the model's prediction
    d = res.decisions
    reps = pd.DataFrame([pairs.loc[s] for s in d["cluster_representatives"]])
    seen = o.data.merge(reps, on=["electrophile", "nucleophile"])
    obs = seen[seen["ligand"].isin(d["multitask_ligands"])].groupby("ligand")["yield"].mean()
    kept = obs[obs.index.isin(d["final_ligands"])]
    dropped = obs[~obs.index.isin(d["final_ligands"])]
    assert dropped.empty or kept.empty or kept.min() >= dropped.max()


@needs_data
def test_general_pipeline_single_substrate_fixes_extra_factors():
    """Shields: base/concentration/temperature stay at the reference; solvent opens only in the final space."""
    o = load_benchmark("shields_arylation")
    lig = KrakenLigandFeaturizer().transform(o.levels("ligand"))
    X = encode_conditions(o.search_space(), o.factors)
    ref = o.data.iloc[0][["base", "concentration", "temperature", "solvent"]].to_dict()
    spec = PipelineSpec(solvent="solvent", fixed=("base", "concentration", "temperature"))
    res = run_overview_pipeline(o, spec, ref, lig, X)
    run = o.data.loc[[r for r, _, _ in res.lab.log]]
    for f in spec.fixed:
        assert (run[f] == ref[f]).all()
    screen = run.iloc[: res.lab.stage_counts()["WP2A"]]
    assert (screen["solvent"] == ref["solvent"]).all() and run["solvent"].nunique() >= 1
    assert res.decisions["final_space_size"] == 12 * 4 and res.lab.stage_counts()["BO"] == 12
    best = res.lab.measured[res.lab.row(**res.recommended)]  # recommendation = best reaction run
    assert best == max(y for _, _, y in res.lab.log)


@needs_data
def test_general_pipeline_metal_screen_uses_controls():
    o = load_benchmark("cernak_cn")
    from catbots import NONE
    key = ["solvent", "base", "temperature", "electrophile", "nucleophile"]
    ref = o.data[o.data["metal_complex"] == NONE][key].iloc[0].to_dict()
    ligs = [l for l in o.levels("ligand") if l != NONE]
    lig = pd.DataFrame(np.random.default_rng(0).normal(size=(len(ligs), 4)), index=ligs)  # any descriptors
    X = encode_conditions(o.search_space(), o.factors)
    spec = PipelineSpec(metal="metal_complex", metal_group="metal", solvent="solvent", fixed=tuple(k for k in key if k != "solvent"))
    small = OverviewBatches(ligand_pool=14, first_ligands=6, final_ligands=4, bo_fraction=0.25)
    res = run_overview_pipeline(o, spec, ref, lig, X, batches=small)
    wp2a = o.data.loc[[r for r, s, _ in res.lab.log if s == "WP2A"]]
    assert NONE in set(wp2a["metal_complex"]) and wp2a["ligand"].nunique() == 6
    assert len(wp2a) == 6 * (len(res.decisions["metal_sources_screened"]) + 1)
    assert set(res.decisions["metal_sources_kept"]) <= set(res.decisions["metal_sources_screened"])
