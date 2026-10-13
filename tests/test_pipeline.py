import numpy as np
import pandas as pd
import pytest

from catbots import NONE, load_benchmark
from catbots.campaign import run_bo, run_random, run_screen_bo, summarise
from catbots.featurizers import KrakenLigandFeaturizer, encode_conditions
from catbots.paths import BENCHMARK_DIR
from catbots.screening import metal_activity, screen_factor, top_levels
from catbots.selection import kennard_stone, select_diverse


def _legacy_kennard_stone(X, k):
    """Verbatim algorithm from src/ligand_screening/kennard_stone_wp2a.py."""
    G = X @ X.T
    H = np.sum(X**2, axis=1, keepdims=True)
    D2 = H + H.T - 2 * G
    np.fill_diagonal(D2, 0.0)
    i1, i2 = np.unravel_index(np.argmax(D2), D2.shape)
    selected = [int(i1), int(i2)]
    min_dist2 = np.minimum(D2[:, i1], D2[:, i2])
    min_dist2[i1] = -np.inf
    min_dist2[i2] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(min_dist2))
        selected.append(nxt)
        min_dist2 = np.minimum(min_dist2, D2[:, nxt])
        min_dist2[nxt] = -np.inf
    return selected


@pytest.mark.parametrize("seed", range(5))
def test_kennard_stone_matches_legacy(seed):
    X = np.random.default_rng(seed).normal(size=(120, 8))
    assert kennard_stone(X, 40) == _legacy_kennard_stone(X, 40)


def test_kennard_stone_initial_and_bounds():
    X = np.random.default_rng(0).normal(size=(30, 3))
    sel = kennard_stone(X, 5, initial=[7])
    assert sel[0] == 7 and len(set(sel)) == 5
    with pytest.raises(ValueError):
        kennard_stone(X, 31)


def test_select_diverse_ranks():
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) ** 2, "name": list("abcdefghij")})
    out = select_diverse(df, 4)
    assert out["ks_rank"].tolist() == [1, 2, 3, 4] and {0, 9} <= set(out.index)


needs_data = pytest.mark.skipif(not BENCHMARK_DIR.exists(), reason="benchmark data not available")


@pytest.fixture(scope="module")
def shields():
    return load_benchmark("shields_arylation")


@needs_data
def test_encode_conditions(shields):
    X = encode_conditions(shields.search_space(), shields.factors)
    assert X.shape == (1728, 12 + 4 + 4 + 1 + 1)
    lig = KrakenLigandFeaturizer().transform(shields.levels("ligand"))
    Xk = encode_conditions(shields.search_space(), shields.factors, descriptors={"ligand": lig})
    assert Xk.shape == (1728, lig.shape[1] + 4 + 4 + 1 + 1) and np.isfinite(Xk).all()


@needs_data
def test_screen_factor_counts_budget(shields):
    shields.reset_budget()
    fixed = shields.data.iloc[0][[f for f in shields.factors if f != "ligand"]].to_dict()
    screen = screen_factor(shields, "ligand", fixed)
    assert len(screen) == 12 and shields.n_queries == 12
    assert len(top_levels(screen, "ligand", 3)) == 3


@needs_data
def test_metal_activity_on_cernak():
    o = load_benchmark("cernak_cn")
    act = metal_activity(o.data, match_on=["ligand", "base", "solvent", "electrophile", "nucleophile", "temperature"])
    assert set(act.index) == {"Pd", "Ni", "Cu"}
    assert act.loc["Pd", "max_effect"] > 50  # Pd reaches ~99% where its control gives far less


@needs_data
@pytest.mark.parametrize("runner", [run_random, run_bo, run_screen_bo])
def test_campaigns_spend_exact_budget(shields, runner):
    X = encode_conditions(shields.search_space(), shields.factors)
    shields.reset_budget()
    res = runner(shields, X, budget=20, seed=0)
    assert len(res.rows) == len(set(res.rows)) == 20 and shields.n_queries == 20
    s = summarise([res], shields)
    assert s.loc[0, "n_experiments"] == 20 and 0 <= s.loc[0, "top_coverage"] <= 1
