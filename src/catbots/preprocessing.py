"""Descriptor pre-processing shared by the pipeline stages."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def drop_correlated(X: pd.DataFrame, threshold: float = 0.7, min_variance: float = 1e-3) -> list[str]:
    """Greedy correlation filter from corr_variance_wp2a.py: min-max normalise, drop near-constant columns,
    then repeatedly remove the lower-variance member of the most correlated pair until all |r| < threshold.

    Returns the kept column names (in input order).
    """
    X = X.select_dtypes(include=[np.number]).dropna(axis=1)
    work = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns, index=X.index)
    var = work.var(ddof=1)
    work = work.loc[:, var > min_variance]
    var = var[work.columns]
    while work.shape[1] > 1:
        corr = work.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        if upper.max().max() < threshold:
            break
        a, b = upper.stack().idxmax()
        drop = a if var[a] < var[b] else b if var[b] < var[a] else max(a, b)
        work = work.drop(columns=[drop])
    return [c for c in X.columns if c in work.columns]


def standardise(X: pd.DataFrame) -> pd.DataFrame:
    """Column-wise z-scores (zero-variance columns left at 0)."""
    Z = StandardScaler().fit_transform(X)
    return pd.DataFrame(np.nan_to_num(Z), columns=X.columns, index=X.index)
