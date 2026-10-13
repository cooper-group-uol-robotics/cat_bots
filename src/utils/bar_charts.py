import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.api.types import is_numeric_dtype
import re
from typing import Iterable, Optional, List, Tuple

# PLOT THE DISTRIBUTION OF DESCRIPTOR VALUES
def plot_descriptor_bars(
    df: pd.DataFrame,
    id_col: str = "MolID",
    exclude_cols: Iterable[str] = ("ks_rank", "CAS Registry Number", "MolID", "CAS Index Name", "Canonical SMILES"),
    output_pdf_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (10, 4),
    rotate_xticks: int = 90,
    sort_by_id: bool = True,
    dropna_rows: bool = True,
) -> List[str]:

    if id_col not in df.columns:
        raise ValueError(f"`{id_col}` column not found in dataframe.")

    candidate_cols = [c for c in df.columns if c not in set(exclude_cols)]
    descriptor_cols = [c for c in candidate_cols if is_numeric_dtype(df[c])]
    if not descriptor_cols:
        print("No numeric descriptor columns found to plot.")
        return []

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

    def _safe(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())

    pdf = PdfPages(output_pdf_path) if output_pdf_path else None
    plotted = []

    for col in descriptor_cols:
        y = work[col]
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
        plt.ylabel(col)
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

    return plotted

df = pd.read_csv("../data/substrates/ks36_culled_abs_minmax.csv")

plotted = plot_descriptor_bars(
    df,
    id_col="MolID",
    exclude_cols=("ks_rank", "CAS Registry Number", "MolID", "CAS Index Name", "Canonical SMILES"),
    output_pdf_path="../data/substrates/descriptor_bars_substrates_ks36_culled_abs_minmax.pdf",
    show=False,
    figsize=(12, 4),
    rotate_xticks=90,
)

print(f"Plotted {len(plotted)} descriptor columns.")