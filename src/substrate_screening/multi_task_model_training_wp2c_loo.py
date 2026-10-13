# train_multitask_elasticnet_LOO.py
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- config ----------------
seed = 42
np.random.seed(seed)
TOP_K = 10

# ---------------- data load (same as before) ----------------
ligand_60 = pd.read_csv("../data/reactions/wp2a_60_ligand_datapoints_filtered.csv")
substrate_96 = pd.read_csv("../data/reactions/wp2c_96_substrate_datapoints_filtered.csv")

for df in (ligand_60, substrate_96):
    df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True, errors="ignore")
    if "SMILES" not in df.columns:
        raise ValueError("Both files must contain a 'SMILES' column.")
    df["SMILES"] = df["SMILES"].astype(str).str.strip()

# 24 ligands overlapping
smiles_24 = set(substrate_96["SMILES"].dropna().unique())
ligand_24 = ligand_60[ligand_60["SMILES"].isin(smiles_24)].copy()

# unify columns and combine
all_cols = sorted(set(ligand_24.columns) | set(substrate_96.columns))
ligand_24   = ligand_24.reindex(columns=all_cols)
substrate_96 = substrate_96.reindex(columns=all_cols)
combined = pd.concat([substrate_96, ligand_24], ignore_index=True)

# ---------------- outputs ----------------
outdir = Path("../data/ML_multitask")
(outdir / "models").mkdir(parents=True, exist_ok=True)
(outdir / "top_features").mkdir(parents=True, exist_ok=True)
(outdir / "importances").mkdir(parents=True, exist_ok=True)
(outdir / "performances").mkdir(parents=True, exist_ok=True)

# ---------------- utilities ----------------
def multi_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average"))
    return {"r2": r2, "mae": mae, "rmse": rmse}

def fit_enet_pipe(X_train, Y_train_scaled, inner_cv):
    """
    Build X-pipeline (impute+scale) + MultiTaskElasticNetCV.
    Y is already standardized per task before calling this.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MultiTaskElasticNetCV(
            l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            alphas=np.logspace(-3, 3, 60),
            cv=inner_cv,
            random_state=seed,
            max_iter=200000,
            tol=1e-3,
            n_jobs=-1,
        ))
    ])
    pipe.fit(X_train, Y_train_scaled)
    return pipe

# ---------------- tidy & define IDs ----------------
df = combined.copy()
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

SUB_CAND = ["substrate", "substrate_SMILES", "SMILES_substrate", "SMILES_sub"]
SUB_COL = next((c for c in SUB_CAND if c in df.columns), None)
if SUB_COL is None:
    SUB_COL = "SMILES" if "SMILES" in df.columns else None
if SUB_COL is None:
    raise ValueError("No substrate ID column found.")

TARGET = "Yield (%)"
if TARGET not in df.columns:
    raise ValueError("Missing 'Yield (%)' column.")

LIG_ID_CAND = ["ligand_SMILES", "ligand", "Ligand", "L_SMILES", "Ligand_SMILES", "SMILES"]
LIG_COL = next((c for c in LIG_ID_CAND if c in df.columns), None)

df[SUB_COL] = df[SUB_COL].astype(str).str.strip()
if LIG_COL:
    df[LIG_COL] = df[LIG_COL].astype(str).str.strip()

# numeric candidate features (exclude target & obvious IDs)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in {TARGET, "No."}]

# ligand-only descriptor columns (constant per ligand across substrates)
if LIG_COL:
    grp = df.groupby(LIG_COL)
else:
    df["_ligand_key_tuple"] = df[num_cols].apply(lambda r: tuple(r.values.tolist()), axis=1)
    LIG_COL = "_ligand_key_tuple"
    grp = df.groupby(LIG_COL)

ligand_cols = [c for c in num_cols if grp[c].nunique(dropna=False).le(1).all()]
if not ligand_cols:
    raise ValueError("No ligand-only numeric descriptors detected.")

# Build ligand x features (X) and ligand x substrates (Y)
X_all = (df.drop_duplicates(subset=[LIG_COL])
           .set_index(LIG_COL)[ligand_cols]
           .apply(pd.to_numeric, errors="coerce"))
Y_all = df.pivot_table(index=LIG_COL, columns=SUB_COL, values=TARGET, aggfunc="mean")

# align, require complete Y across all substrates
common = X_all.index.intersection(Y_all.index)
X_all = X_all.loc[common].copy()
Y_all = Y_all.loc[common].copy()
mask_complete = ~Y_all.isna().any(axis=1)
X_all = X_all.loc[mask_complete]
Y_all = Y_all.loc[mask_complete]

# light winsorization on X (same as your last tweak)
if X_all.shape[0] > 0:
    q_low = X_all.quantile(0.005)
    q_hi  = X_all.quantile(0.995)
    X_all = X_all.clip(lower=q_low, upper=q_hi, axis=1)

n_ligs, n_tasks = len(X_all), Y_all.shape[1]
if n_ligs < 5 or n_tasks < 2:
    raise ValueError(f"Too few ligands/tasks after filtering: ligands={n_ligs}, tasks={n_tasks}")

print(f"Ligands: {n_ligs} | Substrates (tasks): {n_tasks} | Features: {X_all.shape[1]}")

# ---------------- Leave-One-Ligand-Out CV ----------------
loo = LeaveOneOut()
y_true_all, y_pred_all = [], []

# collect per-task metrics fold-by-fold if you want to inspect later (optional)
# per_fold_metrics = []

for train_idx, test_idx in loo.split(X_all):
    X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
    Y_tr, Y_te = Y_all.iloc[train_idx], Y_all.iloc[test_idx]

    # standardize Y per task based on TRAIN only
    Y_mean = Y_tr.mean(axis=0)
    Y_std  = Y_tr.std(axis=0).replace(0, 1.0)
    Y_tr_scaled = (Y_tr - Y_mean) / Y_std

    inner_cv = KFold(n_splits=min(5, len(X_tr)), shuffle=True, random_state=seed)

    # fit on scaled targets
    pipe = fit_enet_pipe(X_tr, Y_tr_scaled, inner_cv=inner_cv)

    # predict (scaled), then invert scaling
    Y_te_pred_scaled = pipe.predict(X_te)
    Y_te_pred = (pd.DataFrame(Y_te_pred_scaled, columns=Y_all.columns, index=Y_te.index) * Y_std) + Y_mean

    y_true_all.append(Y_te.values)
    y_pred_all.append(Y_te_pred.values)

# concatenate LOO predictions/ground truth
Y_true = np.vstack(y_true_all)   # shape (n_ligs, n_tasks)
Y_pred = np.vstack(y_pred_all)   # shape (n_ligs, n_tasks)

# overall (averaged across tasks) metrics
loo_avg = {
    "r2":   r2_score(Y_true, Y_pred, multioutput="uniform_average"),
    "mae":  mean_absolute_error(Y_true, Y_pred, multioutput="uniform_average"),
    "rmse": np.sqrt(mean_squared_error(Y_true, Y_pred, multioutput="uniform_average")),
}

# per-substrate metrics
per_task_rows = []
for j, task in enumerate(Y_all.columns):
    r2  = r2_score(Y_true[:, j], Y_pred[:, j])
    mae = mean_absolute_error(Y_true[:, j], Y_pred[:, j])
    rmse = np.sqrt(mean_squared_error(Y_true[:, j], Y_pred[:, j]))
    per_task_rows.append({"substrate": task, "r2": r2, "mae": mae, "rmse": rmse})

pd.DataFrame([{
    "model": "multitask_enet_LOO",
    "n_features": X_all.shape[1],
    "loo_r2": loo_avg["r2"],
    "loo_mae": loo_avg["mae"],
    "loo_rmse": loo_avg["rmse"],
}]).to_csv(outdir / "performances/metrics_LOO_avg.csv", index=False)

pd.DataFrame(per_task_rows).to_csv(outdir / "performances/metrics_LOO_per_substrate.csv", index=False)

print(f"LOO (avg over substrates): R2={loo_avg['r2']:.3f}, MAE={loo_avg['mae']:.2f}, RMSE={loo_avg['rmse']:.2f}")

# ---------------- Feature importance on ALL data ----------------
# For final feature ranking + models we fit once on ALL ligands.
# We standardize Y per task on ALL data (this affects coef scales across tasks consistently).
Y_mean_all = Y_all.mean(axis=0)
Y_std_all  = Y_all.std(axis=0).replace(0, 1.0)
Y_all_scaled = (Y_all - Y_mean_all) / Y_std_all

inner_cv_all = KFold(n_splits=min(5, len(X_all)), shuffle=True, random_state=seed)
pipe_all = fit_enet_pipe(X_all, Y_all_scaled, inner_cv=inner_cv_all)

# joint importance (ℓ2 across tasks) from scaled model
mt_all = pipe_all.named_steps["model"]   # MultiTaskElasticNetCV
coef_all = mt_all.coef_                  # shape (n_tasks, n_features) in scaled target space
feat_names = X_all.columns.tolist()

l2 = np.sqrt((coef_all ** 2).sum(axis=0))
imp_df = (pd.DataFrame({"feature": feat_names, "l2_norm_across_tasks": l2})
            .sort_values("l2_norm_across_tasks", ascending=False)
            .reset_index(drop=True))
imp_df.to_csv(outdir / "importances/multitask_enet_importance_l2_allData.csv", index=False)

# ---------------- Select top-K & retrain on ALL data ----------------
top_features = imp_df.head(min(TOP_K, len(imp_df)))["feature"].tolist()
pd.Series(top_features, name="feature").to_csv(outdir / "top_features/multitask_enet_selected_top10.csv", index=False)

selector = ColumnTransformer([("sel", "passthrough", top_features)], remainder="drop")
pipe_top = Pipeline([
    ("select", selector),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", MultiTaskElasticNetCV(
        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        alphas=np.logspace(-3, 3, 60),
        cv=inner_cv_all,
        random_state=seed,
        max_iter=200000,
        tol=1e-3,
        n_jobs=-1,
    ))
])

# Fit top-K on scaled Y (ALL data)
pipe_top.fit(X_all[top_features], Y_all_scaled)

# ---------------- Save artifacts ----------------
dump({"pipeline": pipe_all,
      "ligand_features": feat_names,
      "tasks": Y_all.columns.tolist(),
      "y_mean": Y_mean_all.values.astype(float),
      "y_std":  Y_std_all.values.astype(float)},
     outdir / "models/multitask_enet_all.joblib")

dump({"pipeline": pipe_top,
      "ligand_features": top_features,
      "tasks": Y_all.columns.tolist(),
      "y_mean": Y_mean_all.values.astype(float),
      "y_std":  Y_std_all.values.astype(float)},
     outdir / "models/multitask_enet_top10.joblib")

print("\n[OK] Saved:")
print(" - LOO avg metrics :", outdir / "performances/metrics_LOO_avg.csv")
print(" - LOO per-substrate metrics:", outdir / "performances/metrics_LOO_per_substrate.csv")
print(" - importances (all data):", outdir / "importances/multitask_enet_importance_l2_allData.csv")
print(" - top10 list :", outdir / "top_features/multitask_enet_selected_top10.csv")
print(" - models     :", outdir / "models/multitask_enet_all.joblib",
      "and", outdir / "models/multitask_enet_top10.joblib")
print(f"Tasks (substrates): {Y_all.columns.tolist()}")
print(f"Top-10 ligand features (joint across tasks): {top_features}")
