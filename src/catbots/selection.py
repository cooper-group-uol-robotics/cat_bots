"""Kennard-Stone (max-min) selection, shared by every sampling step of the pipeline."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def kennard_stone(X: np.ndarray, k: int, initial: Sequence[int] | None = None) -> list[int]:
    """Indices of ``k`` rows of ``X`` in Kennard-Stone selection order.

    Starts from the farthest pair (or from ``initial`` if given, e.g. points already measured) and then
    repeatedly adds the point whose nearest selected neighbour is farthest away.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if not 0 < k <= n:
        raise ValueError(f"k={k} must be between 1 and n_samples={n}")
    sq = np.sum(X**2, axis=1)

    def d2_to(i: int) -> np.ndarray:  # squared distances to row i, O(n*d) memory
        return np.maximum(sq + sq[i] - 2 * X @ X[i], 0.0)

    if initial is not None and len(initial) > 0:
        selected = [int(i) for i in initial]
    else:  # the farthest pair needs the full matrix; only done when no starting points are given
        D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
        np.fill_diagonal(D2, 0.0)
        i1, i2 = np.unravel_index(np.argmax(D2), D2.shape)
        selected = [int(i1), int(i2)]
        del D2
    if len(selected) >= k:
        return selected[:k]

    min_d2 = np.min([d2_to(i) for i in selected], axis=0)
    min_d2[selected] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(min_d2))
        selected.append(nxt)
        min_d2 = np.minimum(min_d2, d2_to(nxt))
        min_d2[nxt] = -np.inf
    return selected


def select_diverse(df: pd.DataFrame, k: int, features: Sequence[str] | None = None) -> pd.DataFrame:
    """Kennard-Stone on numeric descriptor columns (median-imputed, standardised), as in the WP2 scripts.

    Returns the selected rows in selection order with a 1-based ``ks_rank`` column.
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    if not features:
        raise ValueError("no numeric descriptor columns to select on")
    X = SimpleImputer(strategy="median").fit_transform(df[list(features)])
    X = StandardScaler().fit_transform(X)
    order = kennard_stone(X, k)
    out = df.iloc[order].copy()
    out.insert(0, "ks_rank", range(1, len(out) + 1))
    return out
