# streamlit_ks36_dashboard_flexible_data.py
import sys, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import Draw

# Optional UMAP
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ---------------------- CLI arg (optional) ----------------------
def parse_cli_data_path():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=str, default=None, help="Path to input CSV")
    args, _ = parser.parse_known_args()
    return args.data

CLI_DATA = parse_cli_data_path()


# ---------------------- Streamlit setup ----------------------
st.set_page_config(page_title="KS-36 Diversity Dashboard", layout="wide")
st.title("KS-36 Diversity Dashboard")

# Base dir so relative paths can be resolved against this script
BASE = Path(__file__).resolve().parent
DEFAULT_IN_CSV = CLI_DATA or (BASE / "../data/substrates/ks36_culled_abs_minmax_morgan.csv")


# ---------------------- Helpers ----------------------
def bits_to_array(bitstring: str) -> np.ndarray:
    b = np.frombuffer(bytearray(str(bitstring), "utf-8"), dtype="S1")
    return (b == b"1").astype(np.float32)

@st.cache_data(show_spinner=False)
def read_csv_any(pathlike) -> pd.DataFrame:
    return pd.read_csv(pathlike)

@st.cache_data(show_spinner=False)
def build_feature_matrix(df_in: pd.DataFrame, mode: str):
    """
    mode: 'Morgan fingerprints' OR 'Numeric descriptors'
    Returns (X, feature_names, note_str)
    """
    if mode == "Morgan fingerprints":
        if "morgan_fp_bits" not in df_in.columns:
            raise ValueError("morgan_fp_bits column not found.")
        X = np.vstack([bits_to_array(s) for s in df_in["morgan_fp_bits"].astype(str)])
        feats = [f"fp_{i}" for i in range(X.shape[1])]
        return X, feats, "Using 0/1 Morgan fingerprint bits (no scaling)."

    # Numeric descriptors
    ignore_cols = {
        "CAS Registry Number", "MolID", "CAS Index Name", "Canonical SMILES",
        "ks_rank", "canonical_smiles", "morgan_fp_bits", "SMILES", "smiles", "No."
    }
    drop_cols = [c for c in ignore_cols if c in df_in.columns]
    desc_df = df_in.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    if desc_df.shape[1] == 0:
        raise ValueError("No numeric descriptors found after excluding ID/non-feature columns.")
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X = imp.fit_transform(desc_df.values)
    X = scl.fit_transform(X)
    return X, desc_df.columns.tolist(), "Using all numeric descriptors (median-imputed, standardized)."

def compute_embedding(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "UMAP":
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(X)
    else:
        return PCA(n_components=2, random_state=seed).fit_transform(X)


# ---------------------- Data source controls ----------------------
st.sidebar.header("Data")

data_src = st.sidebar.radio("Choose data source", ["Path", "Upload"], index=0)

df = None
data_note = ""

if data_src == "Path":
    default_str = str(DEFAULT_IN_CSV)
    user_path_str = st.sidebar.text_input("CSV path", default_str).strip()
    if user_path_str:
        p = Path(user_path_str)
        path_resolved = p if p.is_absolute() else (BASE / p)
        try:
            df = read_csv_any(path_resolved)
            data_note = f"Loaded from: `{path_resolved}`"
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            data_note = "Loaded from uploaded file."
        except Exception as e:
            st.sidebar.error(f"Could not read uploaded CSV: {e}")

if df is None:
    st.info("Provide a CSV via the sidebar (path or upload). Trying default …")
    try:
        df = read_csv_any(DEFAULT_IN_CSV)
        data_note = f"Loaded default: `{DEFAULT_IN_CSV}`"
    except Exception as e:
        st.error(f"Could not load default CSV: {e}")
        st.stop()

# Clean unnamed columns
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].copy()

# Find a SMILES column to render molecules
smiles_col = next((c for c in ["canonical_smiles", "Canonical SMILES", "SMILES", "smiles"] if c in df.columns), None)
if smiles_col is None:
    st.error("No SMILES column found (expected one of: canonical_smiles, Canonical SMILES, SMILES, smiles).")
    st.stop()

st.caption(data_note)


# ---------------------- Sidebar: Projection & Clustering ----------------------
st.sidebar.header("Projection & Clustering")

# Determine available feature spaces
available_modes = []
has_fp = "morgan_fp_bits" in df.columns
if has_fp:
    available_modes.append("Morgan fingerprints")

# Check that numeric descriptors are available after dropping IDs
try:
    _Xtmp, _feats, _ = build_feature_matrix(df, "Numeric descriptors")
    has_desc = True
    available_modes.append("Numeric descriptors")
except Exception:
    has_desc = False

if not available_modes:
    st.error("No valid feature space found (neither fingerprints nor numeric descriptors).")
    st.stop()

# Default: prefer fingerprints if present
default_idx = 0
feature_mode = st.sidebar.radio("Feature space", available_modes, index=default_idx)

proj_method = st.sidebar.radio("Projection", ["PCA"] + (["UMAP"] if HAS_UMAP else []))
seed = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)

# Build both spaces (so we can lock axes if user wants)
X_fp = None
X_desc = None
if has_fp:
    try:
        X_fp, _, _ = build_feature_matrix(df, "Morgan fingerprints")
    except Exception:
        has_fp = False
if has_desc:
    try:
        X_desc, _, _ = build_feature_matrix(df, "Numeric descriptors")
    except Exception:
        has_desc = False

# Graceful fallback if user chose fingerprints but not available
if feature_mode == "Morgan fingerprints" and not has_fp:
    st.warning("Morgan fingerprints not found in this CSV (missing 'morgan_fp_bits'). "
               "Falling back to numeric descriptors.")
    feature_mode = "Numeric descriptors"

# Select current X
if feature_mode == "Morgan fingerprints":
    X = X_fp
else:
    X = X_desc

if X is None:
    st.error("Selected feature space is unavailable.")
    st.stop()

# Clusters slider
max_k = max(2, min(12, X.shape[0]))
n_clusters = st.sidebar.slider("KMeans k", min_value=2, max_value=max_k, value=min(8, max_k), step=1)

# Axis locking toggle
lock_axes = st.sidebar.checkbox("Lock axes across modes (for fair comparison)", value=False)


# ---------------------- Projection + Clustering ----------------------
emb_cur = compute_embedding(X, proj_method, seed)

# If locking axes, compute other embedding and union limits (only if both spaces exist)
if lock_axes and (X_fp is not None) and (X_desc is not None):
    X_other = X_desc if feature_mode == "Morgan fingerprints" else X_fp
    emb_other = compute_embedding(X_other, proj_method, seed)
    xmin = float(min(emb_cur[:, 0].min(), emb_other[:, 0].min()))
    xmax = float(max(emb_cur[:, 0].max(), emb_other[:, 0].max()))
    ymin = float(min(emb_cur[:, 1].min(), emb_other[:, 1].min()))
    ymax = float(max(emb_cur[:, 1].max(), emb_other[:, 1].max()))
else:
    xmin, xmax = float(emb_cur[:, 0].min()), float(emb_cur[:, 0].max())
    ymin, ymax = float(emb_cur[:, 1].min()), float(emb_cur[:, 1].max())

# Cluster in current feature space
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
dists = np.linalg.norm(X - centroids[labels], axis=1)


# ---------------------- Layout: scatter + grid ----------------------
c1, c2 = st.columns([2, 1], gap="large")

with c1:
    st.subheader("2D Map")
    plot_df = pd.DataFrame({
        "x": emb_cur[:, 0], "y": emb_cur[:, 1],
        "cluster": labels.astype(int),
        "SMILES": df[smiles_col].astype(str),
        "ks_rank": df.get("ks_rank", pd.Series([None] * len(df)))
    })
    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=80, opacity=0.9, stroke="black", strokeWidth=0.3)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[xmin, xmax]), title="Component 1"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[ymin, ymax]), title="Component 2"),
            color=alt.Color("cluster:N", legend=alt.Legend(title="Cluster")),
            tooltip=["SMILES:N", "cluster:N", alt.Tooltip("ks_rank:Q", title="KS rank")]
        )
        .properties(width="container", height=480, title="2D Map of KS-36")
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Projection: {proj_method}. Feature mode: {feature_mode}. "
        f"Clusters: {n_clusters}. "
        + ("Axes locked across modes." if (lock_axes and (X_fp is not None) and (X_desc is not None)) else "Axes autoscaled to current mode.")
    )

with c2:
    st.subheader("Molecule Grid (white background)")
    info_df = df[[smiles_col]].copy()
    info_df["cluster"] = labels
    info_df["dist_to_centroid"] = dists
    if "ks_rank" in df.columns:
        info_df["ks_rank"] = df["ks_rank"].values

    mode_choice = st.radio(
        "Show",
        ["All (grouped by cluster)", "One cluster", "Representatives (closest to centroid)"],
        index=0
    )
    sel_cluster = None
    if mode_choice == "One cluster":
        sel_cluster = st.number_input("Cluster id", min_value=0, max_value=n_clusters - 1, value=0, step=1)

    # Choose rows to render
    if mode_choice == "Representatives (closest to centroid)":
        reps_idx = info_df.groupby("cluster")["dist_to_centroid"].idxmin().sort_values().tolist()
        grid_df = info_df.loc[reps_idx]
    elif mode_choice == "One cluster":
        grid_df = info_df[info_df["cluster"] == sel_cluster].copy().sort_values("dist_to_centroid")
    else:
        grid_df = info_df.sort_values(["cluster", "dist_to_centroid"]).copy()

    mols = [Chem.MolFromSmiles(s) for s in grid_df[smiles_col].astype(str)]
    legends = []
    for _, row in grid_df.iterrows():
        leg = f"c{int(row['cluster'])} | d={row['dist_to_centroid']:.2f}"
        if "ks_rank" in row and pd.notna(row["ks_rank"]):
            leg = f"KS#{int(row['ks_rank'])} | " + leg
        legends.append(leg)

    if len(mols) == 0:
        st.info("No molecules to display for the current selection.")
    else:
        # PNG ensures white background (good for dark themes)
        img = Draw.MolsToGridImage(
            mols, molsPerRow=3, subImgSize=(220, 220),
            legends=legends, useSVG=False
        )
        st.image(img)


# ---------------------- Footer ----------------------
st.caption(
    "Tip: launch with a path: "
    "`streamlit run streamlit_ks36_dashboard_flexible_data.py -- --data ../data/substrates/ks36_culled_abs_minmax_morgan.csv`"
)
