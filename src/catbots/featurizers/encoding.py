"""Turn a table of reaction conditions into a numeric design matrix for surrogate models."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def encode_conditions(
    space: pd.DataFrame,
    factors: Sequence[str],
    descriptors: Mapping[str, pd.DataFrame] | None = None,
) -> np.ndarray:
    """One-hot encode categorical factors and min-max scale numeric ones.

    ``descriptors`` optionally replaces the one-hot block of a factor with a descriptor table indexed by
    that factor's levels (e.g. ``{"ligand": KrakenLigandFeaturizer().transform(levels)}``); descriptors
    are standardised and must cover every level present in ``space``.
    """
    descriptors = descriptors or {}
    blocks = []
    for f in factors:
        col = space[f]
        if f in descriptors:
            table = descriptors[f]
            levels = col.unique()
            missing = [lv for lv in levels if lv not in table.index or table.loc[lv].isna().any()]
            if missing:
                raise ValueError(f"descriptors for {f!r} missing for {len(missing)} levels, e.g. {missing[0]!r}")
            sub = table.loc[levels]
            sd = sub.std(ddof=0).replace(0, 1.0)
            z = (table - sub.mean()) / sd
            blocks.append(z.loc[col].to_numpy(dtype=float))
        elif pd.api.types.is_numeric_dtype(col):
            lo, hi = col.min(), col.max()
            blocks.append(((col - lo) / (hi - lo if hi > lo else 1.0)).to_numpy(dtype=float)[:, None])
        else:
            blocks.append(pd.get_dummies(col, dtype=float).to_numpy())
    return np.hstack(blocks)
