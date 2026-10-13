import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---- paths ----
active_csv      = "../data/ML_sub/preds/new_preds_logreg_top5_positive_fullrows_ALLDESC.csv"
mt_bundle_path  = "../data/ML_multitask/models/multitask_enet_top10.joblib"  # adjust if different name
out_dir         = Path("../data/ML_multitask/clustering")
out_dir.mkdir(parents=True, exist_ok=True)
out_coef_active = out_dir / "active_substrates_multitask_coeff_embeddings.csv"
out_clusters    = out_dir / "active_substrates_clusters.csv"
out_reps        = out_dir / "active_substrates_cluster_representatives.csv"

K = 8
SEED = 42

bundle = load(mt_bundle_path)
pipe   = bundle["pipeline"]
tasks  = pd.Index(bundle["tasks"]).astype(str)

# recover the top-K ligand feature names actually fed to the multitask model
if "select" in pipe.named_steps:
    # ColumnTransformer([("sel","passthrough", top_features)], ...)
    selector = pipe.named_steps["select"]
    top_features = selector.transformers_[0][2]
else:
    top_features = bundle["ligand_features"]

mt = pipe.named_steps["model"]  # MultiTaskElasticNetCV
coef = mt.coef_                 # shape: (n_tasks, n_features_used)

coef_df = pd.DataFrame(coef, index=tasks, columns=top_features)

active = pd.read_csv(active_csv)
if "SMILES" not in active.columns:
    raise ValueError("Active substrates file must contain a 'SMILES' column.")
active_smiles = active["SMILES"].astype(str).str.strip().unique()

coef_active = coef_df.loc[coef_df.index.intersection(active_smiles)].copy()
if coef_active.empty:
    raise ValueError(
        "None of the active substrates match the multitask model's tasks. "
        "You can only cluster substrates the multitask model was trained on."
    )

X = StandardScaler().fit_transform(coef_active.values)

# k means on substrate embeddings!
k = min(K, len(coef_active))
kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# distance to centroid to pick representative per cluster
dists = np.linalg.norm(X - centroids[labels], axis=1)

# save embeddings with cluster labels
emb = coef_active.copy()
emb["cluster"] = labels
emb["dist_to_centroid"] = dists
emb.to_csv(out_coef_active, index=True)

# choose one representative per cluster (closest to centroid)
rep_idx = []
for c in range(k):
    members = emb[emb["cluster"] == c]
    rep = members["dist_to_centroid"].idxmin()
    rep_idx.append(rep)

reps = emb.loc[rep_idx].copy().sort_values("cluster")

reps_out = reps.reset_index().rename(columns={"index": "SMILES"})[["SMILES", "cluster"]]
reps_out.to_csv(out_reps, index=False)

print("Chosen substrates:", ", ".join(reps_out["SMILES"].astype(str).tolist()))

pd.DataFrame({"SMILES": emb.index, "cluster": emb["cluster"].values}).to_csv(out_clusters, index=False)

print(f"Saved substrate embeddings (+clusters): {out_coef_active}")
print(f"Saved cluster assignments: {out_clusters}")
print(f"Saved one representative per cluster: {out_reps}")
print("Chosen substrates:", ", ".join(reps.index.tolist()))
