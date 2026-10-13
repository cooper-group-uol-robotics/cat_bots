# predict_logreg_top5_untested_with_clustering.py
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

tested_csv   = "../data/reactions/wp2c_36_substrate_datapoints.csv"
all_csv      = "../data/substrates/initial_substrates.csv"

preproc_path = "../data/ML_sub/preproc.joblib"
bundle_path  = "../data/ML_sub/models/logreg_top5.joblib"

out_dir      = Path("../data/ML_sub/preds")
out_main     = out_dir / "new_preds_logreg_top5.csv"
out_pos_min  = out_dir / "new_preds_logreg_top5_positive_only.csv"
out_pos_full = out_dir / "new_preds_logreg_top5_positive_fullrows.csv"
out_pos_full_all = out_dir / "new_preds_logreg_top5_positive_fullrows_ALLDESC.csv"
out_pos_full_clustered = out_dir / "new_preds_logreg_top5_positive_fullrows_clustered.csv"
out_reps     = out_dir / "new_preds_logreg_top5_cluster_reps.csv"

THRESH = 0.5
SEED   = 42
SPECIAL_SMILES = "CC1=CC=C(C=C1)C#C"
K_CLUSTERS = 5

tested   = pd.read_csv(tested_csv)
all_subs = pd.read_csv(all_csv)

for df in (tested, all_subs):
    if "SMILES" in df.columns:
        df["SMILES"] = df["SMILES"].astype(str).str.strip()

if "SMILES" not in tested.columns or "SMILES" not in all_subs.columns:
    raise ValueError("Both input files must contain a 'SMILES' column.")

untested = all_subs[~all_subs["SMILES"].isin(tested["SMILES"])].copy()
untested = untested.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

preproc = load(preproc_path)
bundle  = load(bundle_path)

model             = bundle["model"]
selected_features = bundle["selected_features"]
feature_indices   = bundle["feature_indices"]

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.match(r"^Unnamed")].dropna(axis=1, how="all")

def _align_full_then_slice(df: pd.DataFrame, selected_features, preproc, feature_indices):
    df = _clean_columns(df).copy()

    try:
        train_features = list(preproc.named_steps["imputer"].feature_names_in_)
    except Exception:
        train_features = selected_features

    X_full = df.filter(items=train_features).reindex(columns=train_features)
    X_full = X_full.apply(pd.to_numeric, errors="coerce")
    X_trans = preproc.transform(X_full)

    X_use = X_trans[:, feature_indices]

    X_sel_df = df.filter(items=selected_features).reindex(columns=selected_features)
    X_sel_df = X_sel_df.apply(pd.to_numeric, errors="coerce")

    return X_use, X_sel_df

df_new = untested.copy()
df_new = _clean_columns(df_new).drop(columns=["Yield (%)"], errors="ignore")

id_cols = [c for c in ["No.", "SMILES"] if c in df_new.columns]

# transform & predict (top-5 model)
X_use, X_sel_df = _align_full_then_slice(df_new, selected_features, preproc, feature_indices)
proba_pos = model.predict_proba(X_use)[:, 1]
labels    = (proba_pos >= THRESH).astype(int)

out_dir.mkdir(parents=True, exist_ok=True)
out = df_new[id_cols].copy() if id_cols else pd.DataFrame(index=df_new.index)
out["prob_pos"]   = np.round(proba_pos, 3)
out["pred_label"] = labels
out.to_csv(out_main, index=False)

pos_out = out[out["pred_label"] == 1].copy().sort_values("prob_pos", ascending=False)
pos_out.to_csv(out_pos_min, index=False)

pos_full_all = df_new.loc[pos_out.index].copy()
pos_full_all["prob_pos"]   = pos_out["prob_pos"].values
pos_full_all["pred_label"] = pos_out["pred_label"].values
if "SMILES" in pos_full_all.columns:
    pos_full_all = pos_full_all.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
pos_full_all.to_csv(out_pos_full_all, index=False)

pos_full = (df_new.loc[pos_out.index, id_cols].copy() if id_cols else pd.DataFrame(index=pos_out.index))
pos_full = pos_full.join(X_sel_df.loc[pos_out.index])
pos_full["prob_pos"]   = pos_out["prob_pos"].values
pos_full["pred_label"] = pos_out["pred_label"].values

has_special = ("SMILES" in pos_full.columns) and (pos_full["SMILES"] == SPECIAL_SMILES).any()
if not has_special:
    mask_new = ("SMILES" in df_new.columns) & (df_new["SMILES"] == SPECIAL_SMILES)
    if mask_new.any():
        idx = df_new.index[mask_new][0]
        X_use_row, X_sel_row = _align_full_then_slice(df_new.loc[[idx]], selected_features, preproc, feature_indices)
        p = float(model.predict_proba(X_use_row)[:, 1])

        row_all = df_new.loc[[idx]].copy()
        row_all["prob_pos"]   = round(p, 3)
        row_all["pred_label"] = int(p >= THRESH)
        pos_full_all = pd.concat([pos_full_all, row_all], ignore_index=True)

        row_sel = df_new.loc[[idx], id_cols].copy() if id_cols else pd.DataFrame(index=[idx])
        row_sel = row_sel.join(X_sel_row)
        row_sel["prob_pos"]   = round(p, 3)
        row_sel["pred_label"] = int(p >= THRESH)
        pos_full = pd.concat([pos_full, row_sel], ignore_index=True)
    else:
        fallback = all_subs[all_subs["SMILES"] == SPECIAL_SMILES].copy()
        if not fallback.empty:
            base_row = _clean_columns(fallback.iloc[[0]]).drop(columns=["Yield (%)"], errors="ignore")
            X_use_row, X_sel_row = _align_full_then_slice(base_row, selected_features, preproc, feature_indices)
            p = float(model.predict_proba(X_use_row)[:, 1])

            row_all = base_row.copy()
            row_all["prob_pos"]   = round(p, 3)
            row_all["pred_label"] = int(p >= THRESH)
            pos_full_all = pd.concat([pos_full_all, row_all], ignore_index=True)

            row_sel = base_row[id_cols].copy() if id_cols else pd.DataFrame(index=base_row.index)
            row_sel = row_sel.join(X_sel_row)
            row_sel["prob_pos"]   = round(p, 3)
            row_sel["pred_label"] = int(p >= THRESH)
            pos_full = pd.concat([pos_full, row_sel], ignore_index=True)

if "SMILES" in pos_full_all.columns:
    pos_full_all = pos_full_all.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
if "SMILES" in pos_full.columns:
    pos_full = pos_full.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

pos_full_all.to_csv(out_pos_full_all, index=False)
pos_full.to_csv(out_pos_full, index=False)

# clustering top-5 features
feat_df = pos_full.filter(items=selected_features)
if feat_df.shape[1] == 0:
    raise ValueError("No descriptor columns available for clustering (expected the model's top-5 features).")

imp = SimpleImputer(strategy="median")
scl = StandardScaler()
Xk = imp.fit_transform(feat_df)
Xk = scl.fit_transform(Xk)

k = min(K_CLUSTERS, Xk.shape[0])
kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
labels_k = kmeans.fit_predict(Xk)
centroids = kmeans.cluster_centers_
dists = np.linalg.norm(Xk - centroids[labels_k], axis=1)

pos_full_clust = pos_full.copy()
pos_full_clust["cluster"] = labels_k
pos_full_clust["dist_to_centroid"] = dists
pos_full_clust.to_csv(out_pos_full_clustered, index=False)

# clusters representatives
rep_indices = []
special_idx = None
special_cluster = None

if "SMILES" in pos_full_clust.columns:
    spec_mask = pos_full_clust["SMILES"] == SPECIAL_SMILES
    if spec_mask.any():
        special_idx = int(pos_full_clust.index[spec_mask][0])
        special_cluster = int(pos_full_clust.loc[special_idx, "cluster"])

for c in range(k):
    members = pos_full_clust[pos_full_clust["cluster"] == c]
    if members.empty:
        continue
    if special_cluster is not None and c == special_cluster:
        rep_indices.append(special_idx)
    else:
        rep_indices.append(int(members["dist_to_centroid"].idxmin()))

reps_df = pos_full_clust.loc[sorted(set(rep_indices))].copy().sort_values("cluster")

cols_keep = (["SMILES"] if "SMILES" in reps_df.columns else []) + selected_features
reps_df = reps_df[cols_keep]

out_dir.mkdir(parents=True, exist_ok=True)
reps_df.to_csv(out_reps, index=False)

print(f"Saved full predictions: {out_main}")
print(f"Saved positive-only (IDs+scores): {out_pos_min}")
print(f"Saved positives (IDs + ALL original descriptors): {out_pos_full_all}")
print(f"Saved positives (IDs + top-5 features): {out_pos_full}")
print(f"Saved clustered positives: {out_pos_full_clustered}")
print(f"Saved (SMILES + top-5 descriptors only): {out_reps}")

if "SMILES" in reps_df.columns:
    smiles_list = reps_df["SMILES"].astype(str).tolist()
    print(f"The {len(smiles_list)} chosen substrates are: " + ", ".join(smiles_list))
else:
    print("Saved representatives, but SMILES column not found.")
