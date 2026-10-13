import numpy as np
import pandas as pd
import pytest

from catbots import NONE, ReactionOracle, list_benchmarks, load_benchmark
from catbots.featurizers import KrakenLigandFeaturizer
from catbots.paths import BENCHMARK_DIR

pytestmark = pytest.mark.skipif(not BENCHMARK_DIR.exists(), reason="benchmark data not available")

EXPECTED = {
    # name: (n reactions, complete grid?, {factor: n levels})
    "shields_arylation": (1728, True, {"ligand": 12, "base": 4, "solvent": 4, "concentration": 3, "temperature": 3}),
    "doyle_arylation": (1536, True, {"ligand": 24, "electrophile": 8, "nucleophile": 8}),
    "perera_suzuki": (5760, False, {"electrophile": 7, "nucleophile": 4, "ligand": 12, "base": 8, "solvent": 4}),
    "perera_suzuki_edbo": (3696, True, {"electrophile": 4, "nucleophile": 3, "ligand": 11, "base": 7, "solvent": 4}),
}


@pytest.fixture(scope="module")
def oracles():
    return {name: load_benchmark(name) for name in list_benchmarks()}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_grid_shapes(oracles, name):
    n, complete, levels = EXPECTED[name]
    o = oracles[name]
    assert len(o) == n
    assert o.is_complete_grid is complete
    assert {f: o.data[f].nunique() for f in levels} == levels


def test_perera_internal_standard_removed(oracles):
    assert not oracles["perera_suzuki"].data["electrophile"].str.contains("CCc1cccc(CC)c1", regex=False).any()


def test_perera_edbo_matches_shields_copy(oracles):
    """The EDBO subset must reproduce the yields of the Shields-distributed copy (same experiments)."""
    shields = pd.read_csv(BENCHMARK_DIR / "perera_suzuki" / "shields_suzuki.csv")
    edbo = oracles["perera_suzuki_edbo"].data
    assert np.allclose(np.sort(shields["yield"].to_numpy()), np.sort(edbo["yield"].to_numpy()), atol=0.01)


def test_cernak_cn(oracles):
    o = oracles["cernak_cn"]
    assert set(o.data["metal"]) == {"Pd", "Ni", "Cu", NONE}
    assert o.data["n_replicates"].sum() == 33792  # every pair-1 measurement accounted for
    assert o.data["yield"].between(0, 100).all()
    assert not o.data["ligand_name"].str.contains("â€", regex=False).any()


def test_query_and_budget(oracles):
    o = oracles["shields_arylation"]
    o.reset_budget()
    picks = o.search_space().sample(5, random_state=0)
    expected = o.data.loc[picks.index, "yield"].to_numpy()
    assert np.allclose(o.query(picks), expected)
    assert o.n_queries == 5
    with pytest.raises(KeyError):
        o.query({**picks.iloc[0].to_dict(), "temperature": 999.0})


def test_subset(oracles):
    o = oracles["cernak_cn"].subset(metal="Pd")
    assert set(o.data["metal"]) == {"Pd"}
    assert isinstance(o, ReactionOracle)


@pytest.mark.parametrize("name,expected_matched", [
    ("shields_arylation", 12), ("doyle_arylation", 24), ("perera_suzuki", 8)])
def test_kraken_coverage(oracles, name, expected_matched):
    feat = KrakenLigandFeaturizer("reduced")
    ligands = [s for s in oracles[name].levels("ligand") if s != NONE]
    assert feat.matched(ligands).sum() == expected_matched
    X = feat.transform(ligands)
    assert X.shape == (len(ligands), len(feat.columns))
    assert X.dropna().shape[0] == expected_matched
