"""Small RDKit helpers shared by loaders and featurisers."""

from collections.abc import Iterable

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def _is_missing(smiles) -> bool:
    return smiles is None or (isinstance(smiles, float) and pd.isna(smiles)) or str(smiles).strip() == ""


def canonical_smiles(smiles, isomeric: bool = True) -> str | None:
    """RDKit canonical SMILES; strings RDKit cannot parse (e.g. ``~`` bonds) are returned stripped."""
    if _is_missing(smiles):
        return None
    s = str(smiles).strip()
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric) if mol is not None else s


def inchikey14(smiles) -> str | None:
    """First InChIKey block (connectivity only), so stereo and tautomer notation differences still match."""
    if _is_missing(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    key = Chem.MolToInchiKey(mol)
    return key.split("-")[0] if key else None


def remove_fragments(smiles, fragments: Iterable[str]) -> str | None:
    """Drop '.'-separated fragments that match any of ``fragments`` (compared as canonical SMILES)."""
    if _is_missing(smiles):
        return None
    drop = {canonical_smiles(f) for f in fragments}
    kept = [p for p in str(smiles).split(".") if canonical_smiles(p) not in drop]
    return ".".join(kept)


def map_unique(series: pd.Series, fn) -> pd.Series:
    """Apply ``fn`` once per unique value; much faster than ``Series.map(fn)`` on large reaction tables."""
    uniques = series.dropna().unique()
    return series.map({u: fn(u) for u in uniques})
