import pandas as pd

from catbots.chem import canonical_smiles, map_unique
from catbots.oracle import NONE


def smiles_column(series: pd.Series) -> pd.Series:
    """Canonicalise a SMILES column and encode absent components as NONE."""
    return map_unique(series, canonical_smiles).fillna(NONE)


def fix_mojibake(text):
    """Repair UTF-8 text that was decoded as cp1252 (e.g. 'â€²' -> '′')."""
    if not isinstance(text, str):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
