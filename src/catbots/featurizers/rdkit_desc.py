"""RDKit 2D descriptors, used where no DFT substrate descriptors exist (e.g. the arylation benchmarks)."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def rdkit_descriptors(smiles: Iterable[str]) -> pd.DataFrame:
    """All RDKit 2D descriptors, indexed by the input SMILES; columns with any NaN/inf are dropped."""
    smiles = list(dict.fromkeys(smiles))
    rows = {}
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"cannot parse SMILES {s!r}")
        rows[s] = Descriptors.CalcMolDescriptors(mol)
    df = pd.DataFrame.from_dict(rows, orient="index").astype(float)
    return df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
