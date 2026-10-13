import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.api.types import is_numeric_dtype
import re
from typing import Iterable, Optional, List, Tuple


# PLOT THE DISTRIBUTION OF DESCRIPTOR VALUES AFTER NORMALIZATION (MINMAX + ABS)
def plot_descriptor_bars(
    df: pd.DataFrame,
    id_col: str = "No.",
    exclude_cols: Iterable[str] = ("No.", "SMILES", "CAS", "location"),
    output_pdf_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (10, 4),
    rotate_xticks: int = 90,
    sort_by_id: bool = True,
    dropna_rows: bool = True,
    normalize: bool = True,
    use_abs: bool = True,
    robust_clip: Optional[Tuple[float, float]] = None,

    return_transformed_df: bool = True,
    output_df_csv_path: Optional[str] = None,
):
    if id_col not in df.columns:
        raise ValueError(f"`{id_col}` column not found in dataframe.")

    exclude_set = set(exclude_cols)
    candidate_cols = [c for c in df.columns if c not in exclude_set]
    descriptor_cols = [c for c in candidate_cols if is_numeric_dtype(df[c])]
    if not descriptor_cols:
        print("No numeric descriptor columns found to plot.")
        return [], df.copy() if return_transformed_df else None

    work = df.copy()

    if sort_by_id:
        try:
            work = work.assign(__id_tmp__=pd.to_numeric(work[id_col], errors="raise")).sort_values("__id_tmp__")
            x_vals = work[id_col].astype(str).tolist()
            work = work.drop(columns="__id_tmp__", errors="ignore")
        except Exception:
            work = work.sort_values(id_col)
            x_vals = work[id_col].astype(str).tolist()
    else:
        x_vals = work[id_col].astype(str).tolist()

    def _abs_minmax_01(s: pd.Series) -> pd.Series:
        y = s.astype(float).copy()
        if use_abs:
            y = y.abs()

        if robust_clip is not None:
            ql, qh = robust_clip
            try:
                lo = np.nanquantile(y, ql)
                hi = np.nanquantile(y, qh)
            except Exception:
                lo, hi = np.nanmin(y.values), np.nanmax(y.values)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                y = y.clip(lower=lo, upper=hi)

        ymin = np.nanmin(y.values)
        ymax = np.nanmax(y.values)
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax == ymin:
            return pd.Series(np.zeros_like(y), index=y.index, dtype=float)
        return (y - ymin) / (ymax - ymin)

    df_transformed = work.copy()
    if normalize:
        for col in descriptor_cols:
            df_transformed[col] = _abs_minmax_01(df_transformed[col])
    else:
        if use_abs:
            for col in descriptor_cols:
                df_transformed[col] = df_transformed[col].astype(float).abs()

    if output_df_csv_path:
        df_transformed.to_csv(output_df_csv_path, index=False)

    pdf = PdfPages(output_pdf_path) if output_pdf_path else None
    plotted = []

    for col in descriptor_cols:
        y = df_transformed[col]
        if dropna_rows:
            mask = y.notna()
            x_plot = [x for x, keep in zip(x_vals, mask) if keep]
            y_plot = y[mask].values
        else:
            x_plot = x_vals
            y_plot = y.values

        if len(y_plot) == 0:
            continue

        plt.figure(figsize=figsize)
        plt.bar(x_plot, y_plot)
        plt.xlabel(id_col)
        ylabel = f"{col}"
        if normalize and use_abs:
            ylabel += " (|x|→0–1)"
        elif normalize:
            ylabel += " (0–1)"
        elif use_abs:
            ylabel += " (|x|)"
        plt.ylabel(ylabel)
        plt.title(f"{col} by {id_col}")
        plt.xticks(rotation=rotate_xticks)
        plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()

        if pdf:
            pdf.savefig()
            plt.close()
        else:
            if show:
                plt.show()

        plotted.append(col)

    if pdf:
        pdf.close()
        print(f"Saved {len(plotted)} plots to: {output_pdf_path}")

    return (plotted, df_transformed if return_transformed_df else None)


if __name__ == "__main__":
    df = pd.read_csv("../data/ligands/final_list.csv")

    plotted, df_norm = plot_descriptor_bars(
        df,
        id_col="No.",
        exclude_cols=("No.", "SMILES", "CAS", "location"),
        output_pdf_path="../data/ligands/descriptor_bars_ligands_abs_minmax.pdf",
#        output_df_csv_path="../data/substrates/ks36_culled_abs_minmax.csv",
        show=False,
        figsize=(12, 4),
        rotate_xticks=90,
        normalize=True,
        use_abs=True,
        robust_clip=None,
        return_transformed_df=True
    )

    print(f"Plotted {len(plotted)} descriptor columns.")
