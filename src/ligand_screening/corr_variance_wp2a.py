import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Literal, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors

def drop_corr_greedy_keep_high_var(
    df: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
    corr_threshold: float = 0.70,
    min_variance: float = 0.0,
    numeric_only: bool = True,
    exclude_names: Iterable[str] = (),
    exclude_startswith: Iterable[str] = (),
    exclude_endswith: Iterable[str] = (),
    exclude_contains: Iterable[str] = (),
    exclude_regex: Optional[str] = None,
    impute_missing: bool = False,
    impute_strategy: str = "median",
    use_abs: bool = False,
    normalize: bool = True,
    norm_method: Literal["minmax", "zscore"] = "minmax",
    variance_basis: Literal["normalized", "original"] = "normalized",
    drop_missing_rows: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    cols_index = pd.Index(df.columns)

    def build_exclude_mask(index: pd.Index) -> pd.Series:
        mask = pd.Series(False, index=index, dtype=bool)
        if exclude_names:
            mask |= index.isin(list(exclude_names))
        if exclude_startswith:
            mask |= index.str.startswith(tuple(exclude_startswith), na=False)
        if exclude_endswith:
            mask |= index.str.endswith(tuple(exclude_endswith), na=False)
        for s in exclude_contains or ():
            mask |= index.str.contains(s, regex=False, na=False)
        if exclude_regex:
            mask |= index.str.contains(exclude_regex, regex=True, na=False)
        return mask

    exclude_mask_global = build_exclude_mask(cols_index)

    if cols is None:
        cand_df = df.select_dtypes(include=[np.number]).copy() if numeric_only else df.copy()
    else:
        cand_df = df[list(cols)].copy()

    cand_df = cand_df.loc[:, ~cand_df.columns.isin(cols_index[exclude_mask_global])]
    cand_df = cand_df.dropna(axis=1, how="all")
    if cand_df.shape[1] == 0:
        raise ValueError("No candidate descriptor columns after excluding/dropping empty columns.")

    if drop_missing_rows:
        cand_df = cand_df.dropna(axis=0, how="any")
        if cand_df.shape[0] == 0:
            raise ValueError("No rows left after dropping rows with missing values in candidate columns.")
        df_row_filtered = df.loc[cand_df.index]
    else:
        df_row_filtered = df

    cand_df_original = cand_df.copy()
    work_df = cand_df.astype(float).copy()

    if impute_missing:
        imp = SimpleImputer(strategy=impute_strategy)
        work_df.iloc[:, :] = imp.fit_transform(work_df.values)

    if use_abs:
        work_df = work_df.abs()

    if normalize:
        if norm_method == "minmax":
            scaler = MinMaxScaler()
        elif norm_method == "zscore":
            scaler = StandardScaler()
        else:
            raise ValueError("norm_method must be 'minmax' or 'zscore'")
        work_df.iloc[:, :] = scaler.fit_transform(work_df.values)

    if variance_basis == "normalized":
        variance_vec = work_df.var(axis=0, ddof=1)
    elif variance_basis == "original":
        variance_vec = cand_df_original.var(axis=0, ddof=1)
    else:
        raise ValueError("variance_basis must be 'normalized' or 'original'")

    low_var_cols = variance_vec.index[variance_vec <= min_variance].tolist()
    work_df = work_df.drop(columns=low_var_cols, errors="ignore")
    variance_vec = variance_vec.drop(index=low_var_cols, errors="ignore")

    dropped_rows = []
    for c in low_var_cols:
        dropped_rows.append({
            "column": c,
            "variance": float(variance_vec.get(c, np.nan)),
            "dropped_because": f"variance<=min_variance ({min_variance})",
            "paired_with": None
        })

    if work_df.shape[1] <= 1:
        kept_cols = list(work_df.columns)
        kept_df = pd.DataFrame({"column": kept_cols,
                                "variance": variance_vec.reindex(kept_cols).astype(float).values})
        dropped_df = pd.DataFrame(dropped_rows)
        non_desc_cols = [c for c in df_row_filtered.columns if c not in cand_df.columns]
        df_reduced = df_row_filtered[non_desc_cols + kept_cols].copy()
        corr_abs_final = work_df.corr().abs() if work_df.shape[1] > 0 else pd.DataFrame()
        return df_reduced, kept_df, dropped_df, corr_abs_final

    while True:
        corr_abs = work_df.corr(method="pearson").abs()
        if corr_abs.shape[0] <= 1:
            break
        upper = corr_abs.where(np.triu(np.ones(corr_abs.shape, dtype=bool), k=1))
        max_corr = upper.max().max()
        if pd.isna(max_corr) or max_corr < corr_threshold:
            break
        feature1, feature2 = upper.stack().idxmax()
        v1 = float(variance_vec.get(feature1, np.nan))
        v2 = float(variance_vec.get(feature2, np.nan))

        def tie_break(a: str, b: str) -> str:
            na_a = 0 if impute_missing else work_df[a].isna().sum()
            na_b = 0 if impute_missing else work_df[b].isna().sum()
            if na_a != na_b:
                return a if na_a > na_b else b
            return max(a, b)

        if np.isnan(v1) or np.isnan(v2):
            to_remove = tie_break(feature1, feature2)
        else:
            if v1 < v2:
                to_remove = feature1
            elif v2 < v1:
                to_remove = feature2
            else:
                to_remove = tie_break(feature1, feature2)

        dropped_rows.append({
            "column": to_remove,
            "variance": float(variance_vec.get(to_remove, np.nan)),
            "dropped_because": f"|corr|>={corr_threshold} with {feature2 if to_remove==feature1 else feature1}",
            "paired_with": feature2 if to_remove == feature1 else feature1,
        })

        work_df = work_df.drop(columns=[to_remove])
        variance_vec = variance_vec.drop(index=[to_remove], errors="ignore")
        if work_df.shape[1] <= 1:
            break

    kept_cols = list(work_df.columns)
    kept_df = pd.DataFrame({
        "column": kept_cols,
        "variance": variance_vec.reindex(kept_cols).astype(float).values
    }).sort_values("column").reset_index(drop=True)

    dropped_df = pd.DataFrame(dropped_rows).reset_index(drop=True)
    corr_abs_final = work_df.corr().abs()

    non_cand_cols = [c for c in df_row_filtered.columns if c not in cand_df.columns]
    df_reduced = df_row_filtered[non_cand_cols + kept_cols].copy()

    return df_reduced, kept_df, dropped_df, corr_abs_final


df = pd.read_csv("../data/ligands/kraken_no_delta.csv",
                 usecols=lambda c: not str(c).startswith("Unnamed"))

df_reduced, kept, dropped, corrM = drop_corr_greedy_keep_high_var(
    df,
    corr_threshold=0.70,
    min_variance=1e-3,
    exclude_names=["SMILES", "No."],
    impute_missing=False,
#    impute_strategy="median",
    use_abs=False,
    normalize=True,
    norm_method="minmax",
    variance_basis="normalized",
    drop_missing_rows=True
)

print(f"Kept: {len(kept)} descriptors")
print("Kept:\n", kept.head(70))
print("\nDropped:\n", dropped)

df_reduced.to_csv('../data/ligands/kraken_reduced_0.7_no_delta.csv', index=False)

print(df_reduced.head(n=10))

corrM.to_csv("../data/ligands/corr_matrix_kraken_reduced_0.7_no_delta.csv", index=True)