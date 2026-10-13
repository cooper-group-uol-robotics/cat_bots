import dataclasses

import numpy as np
import pandas as pd
import pytest

from catbots import load_benchmark
from catbots.featurizers import KrakenLigandFeaturizer, encode_conditions
from catbots.paths import BENCHMARK_DIR
from catbots.pipeline import PipelineSpec, run_overview_pipeline
from catbots.prescreen import PrescreenBatches, d_optimal, select_reference


def test_d_optimal_design_is_full_rank_and_unique():
    combos = pd.DataFrame([(a, b, c) for a in "wxyz" for b in "pqr" for c in "uv"], columns=["A", "B", "C"])
    X = np.hstack([np.ones((len(combos), 1)), pd.get_dummies(combos, dtype=float).to_numpy()])
    idx = d_optimal(X, 8, np.random.default_rng(0))
    assert len(set(idx)) == 8
    # 1 + 3 + 2 + 1 = 7 estimable main-effect parameters, so 8 runs must give full rank
    assert np.linalg.matrix_rank(X[idx]) == 7


needs_data = pytest.mark.skipif(not BENCHMARK_DIR.exists(), reason="benchmark data not available")


@needs_data
def test_wp0_on_shields_chooses_reference_and_feeds_pipeline():
    o = load_benchmark("shields_arylation")
    lig = KrakenLigandFeaturizer().transform(o.levels("ligand"))
    spec = PipelineSpec(solvent="solvent", fixed=("base", "concentration", "temperature"),
                        screen=("base", "concentration", "temperature"))
    base_ref = o.data.iloc[0][["base", "concentration", "temperature", "solvent"]].to_dict()
    pre, lab = select_reference(o, spec, base_ref, lig, seed=1)
    assert lab.n_experiments == 30 and lab.stage_counts() == {"WP0": 30}
    assert len(pre.probes) == 3 and pre.design.shape == (10, 4)  # base, conc, T + solvent
    assert set(pre.chosen) == {"base", "concentration", "temperature", "solvent"}
    key = tuple(str(pre.chosen[f]) for f in pre.predicted.index.names)
    assert pre.predicted[key] == pre.predicted.max() or (pre.observed < 5).all(axis=1).any()
    X = encode_conditions(o.search_space(), o.factors)
    res = run_overview_pipeline(o, spec, pre.reference, lig, X, lab=lab)
    counts = res.lab.stage_counts()
    assert counts["WP0"] == 30 and res.n_experiments == o.n_queries == len({r for r, _, _ in res.lab.log})
    for f in spec.fixed:  # every pipeline experiment after WP0 uses the chosen reference levels
        later = o.data.loc[[r for r, s, _ in res.lab.log if s != "WP0"], f]
        assert (later == pre.reference[f]).all()


@needs_data
def test_wp0_requires_screen_factors():
    o = load_benchmark("shields_arylation")
    with pytest.raises(ValueError):
        select_reference(o, PipelineSpec(solvent="solvent", fixed=("base", "concentration", "temperature")),
                         {}, KrakenLigandFeaturizer().transform(o.levels("ligand")))
