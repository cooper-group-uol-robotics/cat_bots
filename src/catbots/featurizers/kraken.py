"""Kraken DFT descriptors for monodentate P(III) ligands, matched by InChIKey connectivity block."""

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from catbots.chem import inchikey14, map_unique
from catbots.paths import LIGAND_DIR

DESCRIPTOR_SETS = {
    # correlation-screened (|r| >= 0.7 greedy, keep higher variance) output of corr_variance_wp2a.py
    "reduced": "kraken_reduced_0.7_no_delta.csv",
    "no_delta": "kraken_no_delta.csv",
    "full": "kraken.csv",
}
_ID_COLS = {"No.", "SMILES", "ID"}


@lru_cache(maxsize=None)
def _load_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: not str(c).startswith("Unnamed"))
    df["_key"] = map_unique(df["SMILES"], inchikey14)
    desc = [c for c in df.columns if c not in _ID_COLS | {"_key"} and pd.api.types.is_numeric_dtype(df[c])]
    # stereoisomers share a connectivity key; average their descriptors
    return df.dropna(subset=["_key"]).groupby("_key")[desc].mean()


class KrakenLigandFeaturizer:
    """Map ligand SMILES to Kraken descriptors.

    Kraken covers monodentate phosphines only, so bidentates, NHCs, N/O ligands
    and "no ligand" come back as all-NaN rows; check ``coverage`` before modelling.
    """

    def __init__(self, descriptor_set: str = "reduced", path: str | Path | None = None):
        if path is None:
            if descriptor_set not in DESCRIPTOR_SETS:
                raise ValueError(f"descriptor_set must be one of {list(DESCRIPTOR_SETS)}")
            path = LIGAND_DIR / DESCRIPTOR_SETS[descriptor_set]
        self.descriptor_set = descriptor_set
        self.table = _load_table(str(path))

    @property
    def columns(self) -> list[str]:
        return self.table.columns.tolist()

    def transform(self, smiles: Iterable[str]) -> pd.DataFrame:
        smiles = pd.Series(list(smiles), dtype=object)
        keys = map_unique(smiles, inchikey14)
        out = self.table.reindex(keys.to_numpy())
        out.index = smiles.to_numpy()
        return out

    def matched(self, smiles: Iterable[str]) -> np.ndarray:
        keys = map_unique(pd.Series(list(smiles), dtype=object), inchikey14)
        return keys.isin(self.table.index).to_numpy()

    def coverage(self, smiles: Iterable[str]) -> float:
        return float(self.matched(smiles).mean())
