import numpy as np
import pandas as pd
import pytest

from catbots import load_benchmark
from catbots.featurizers.chemeleon import DEFAULT_PATH, CheMeleonFeaturizer
from catbots.plotting import plot_curves, plot_final_comparison, plot_stage_trace

needs_cache = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="CheMeleon embedding cache not computed")


@needs_cache
def test_chemeleon_covers_doyle_and_pca_is_compact():
    cm = CheMeleonFeaturizer()
    o = load_benchmark("doyle_arylation")
    ligands = o.levels("ligand")
    E = cm.transform(ligands)
    assert E.shape == (24, 2048) and E.notna().all().all()
    P = cm.transform_pca(ligands, variance=0.9)
    assert 1 <= P.shape[1] <= 23 and list(P.index) == ligands
    with pytest.raises(ValueError):
        cm.transform_pca(["C" * 40 + "[Xe]"])  # not in the cache


@needs_cache
def test_chemeleon_matches_ligands_without_inchikey():
    """Ferrocenyl phosphines written with dative bonds have no InChIKey; they must still match (by SMILES)."""
    from catbots.chem import inchikey14
    o = load_benchmark("cernak_cn")
    no_key = [l for l in o.levels("ligand") if "->[Fe+2]" in l]
    assert no_key and all(inchikey14(l) is None for l in no_key)
    assert CheMeleonFeaturizer().transform(no_key).notna().all(axis=1).all()


def test_plots_write_png_and_csv(tmp_path):
    rng = np.random.default_rng(0)
    runs = {"pipeline": [rng.random(30).cumsum() for _ in range(5)], "random": [rng.random(25).cumsum() for _ in range(5)]}
    table = plot_curves(runs, tmp_path / "c.png", "t", "y", stage_bounds={"A": 10, "B": 30},
                        reference_lines={"ref": 5.0})
    assert (tmp_path / "c.png").exists() and (tmp_path / "c.csv").exists() and "Random mean" in table
    plot_stage_trace(rng.random(20) * 100, ["A"] * 8 + ["B"] * 12, tmp_path / "s.png", "t")
    plot_final_comparison({"pipeline": rng.random(10), "bo": rng.random(10)}, tmp_path / "f.png", "t", "y")
    assert pd.read_csv(tmp_path / "f.csv").shape[0] == 2 and (tmp_path / "s.csv").exists()
