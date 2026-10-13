import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer

seed = 42
np.random.seed(seed)
TOP_K = 10

#this is the dataset of 60 ligands against 1 substrate
ligand_60 = pd.read_csv("../data/reactions/wp2a_60_ligand_datapoints_filtered.csv")

#this is the dataset of 4 substrates x 24 ligands - should contain Yield (%) and SMILES of ligands
substrate_96 = pd.read_csv("../data/reactions/wp2c_96_substrate_datapoints_filtered.csv")

for df in (ligand_60, substrate_96):
    df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True, errors="ignore")
    if "SMILES" not in df.columns:
        raise ValueError("Both files must contain a 'SMILES' column.")
    df["SMILES"] = df["SMILES"].astype(str).str.strip()

# The 24 ligands (unique SMILES) from the 4-substrate dataset
smiles_24 = set(substrate_96["SMILES"].dropna().unique())

# Filter the 60-ligand single-substrate dataset down to those 24 ligands
ligand_24 = ligand_60[ligand_60["SMILES"].isin(smiles_24)].copy()

all_cols = sorted(set(ligand_24.columns) | set(substrate_96.columns))
ligand_24  = ligand_24.reindex(columns=all_cols)
substrate_96 = substrate_96.reindex(columns=all_cols)

# this dataset should be ligand descriptors, substrate SMILES and Yield (%) only
combined = pd.concat([substrate_96, ligand_24], ignore_index=True)

outdir = Path("../data/ML_multitask")
(outdir / "models").mkdir(parents=True, exist_ok=True)
(outdir / "top_features").mkdir(parents=True, exist_ok=True)
(outdir / "importances").mkdir(parents=True, exist_ok=True)
(outdir / "performances").mkdir(parents=True, exist_ok=True)

def multi_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average"))
    return {"r2": r2, "mae": mae, "rmse": rmse}

r2_scorer = make_scorer(r2_score, greater_is_better=True, multioutput="uniform_average")
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False, multioutput="uniform_average")

def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average"))
rmse_scorer = make_scorer(rmse_func, greater_is_better=False)

def cv_means(pipe, X_df, Y_df, n_splits=5):
    cv = KFold(n_splits=min(n_splits, len(X_df)), shuffle=True, random_state=seed)
    scores = cross_validate(
        pipe, X_df, Y_df, cv=cv,
        scoring={"r2": r2_scorer, "neg_mae": mae_scorer, "neg_rmse": rmse_scorer},
        return_train_score=False, n_jobs=None
    )
    return {
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_mae_mean": float(-scores["test_neg_mae"].mean()),
        "cv_rmse_mean": float(-scores["test_neg_rmse"].mean()),
    }

df = combined.copy()
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

# substrate ID column (used to make Y columns)
SUB_CAND = ["substrate", "substrate_SMILES", "SMILES_substrate", "SMILES_sub"]
SUB_COL = next((c for c in SUB_CAND if c in df.columns), None)
if SUB_COL is None:
    if "SMILES" in df.columns:
        SUB_COL = "SMILES"
    else:
        raise ValueError("No substrate ID column found.")

TARGET = "Yield (%)"
if TARGET not in df.columns:
    raise ValueError("Missing 'Yield (%)'.")

# Optional ligand ID
LIG_ID_CAND = ["ligand_SMILES", "ligand", "Ligand", "L_SMILES", "Ligand_SMILES", "SMILES"]
LIG_COL = next((c for c in LIG_ID_CAND if c in df.columns), None)

df[SUB_COL] = df[SUB_COL].astype(str).str.strip()
if LIG_COL:
    df[LIG_COL] = df[LIG_COL].astype(str).str.strip()

# numeric candidates excluding obvious non-features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in {TARGET, "No."}]

# Identify ligand-only descriptor columns (constant within each ligand across substrates)
if LIG_COL:
    grp = df.groupby(LIG_COL)
else:
    df["_ligand_key_tuple"] = df[num_cols].apply(lambda r: tuple(r.values.tolist()), axis=1)
    LIG_COL = "_ligand_key_tuple"
    grp = df.groupby(LIG_COL)

ligand_cols = [c for c in num_cols if grp[c].nunique(dropna=False).le(1).all()]
if not ligand_cols:
    raise ValueError("No ligand-only numeric descriptors detected.")

# X: one row per ligand (descriptors), Y: wide matrix (ligand × substrates)
X_all = (df.drop_duplicates(subset=[LIG_COL])
           .set_index(LIG_COL)[ligand_cols]
           .apply(pd.to_numeric, errors="coerce"))
Y_all = df.pivot_table(index=LIG_COL, columns=SUB_COL, values=TARGET, aggfunc="mean")

# align & drop ligands with any missing substrate yields
common = X_all.index.intersection(Y_all.index)
X_all = X_all.loc[common].copy()
Y_all = Y_all.loc[common].copy()

mask_complete = ~Y_all.isna().any(axis=1)
X_all = X_all.loc[mask_complete]
Y_all = Y_all.loc[mask_complete]
q_low = X_all.quantile(0.005)
q_hi  = X_all.quantile(0.995)
X_all = X_all.clip(lower=q_low, upper=q_hi, axis=1)

n_ligs, n_tasks = len(X_all), Y_all.shape[1]
if n_ligs < 5 or n_tasks < 2:
    raise ValueError(f"Too few ligands/tasks after filtering: ligands={n_ligs}, tasks={n_tasks}")

idx = np.arange(n_ligs)
train_idx, temp_idx = train_test_split(idx, test_size=0.20, random_state=seed)
test_idx, val_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed)

X_train, Y_train = X_all.iloc[train_idx], Y_all.iloc[train_idx]
X_test,  Y_test  = X_all.iloc[test_idx],  Y_all.iloc[test_idx]
X_val,   Y_val   = X_all.iloc[val_idx],   Y_all.iloc[val_idx]

# All features model
inner_cv = KFold(n_splits=min(5, len(X_train)), shuffle=True, random_state=seed)

pipe_all = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", MultiTaskElasticNetCV(
        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        alphas=np.logspace(-3, 3, 60),  # stronger & wider regularization grid
        cv=inner_cv,
        random_state=seed,
        max_iter=200000,  # more iterations
        tol=1e-3,  # slightly looser tolerance
        n_jobs=-1,
    ))
])

# CV on TRAIN (5-fold mean)
cv_all = cv_means(pipe_all, X_train, Y_train, n_splits=5)

# Fit on TRAIN and evaluate on TEST + VAL
pipe_all.fit(X_train, Y_train)
Y_test_pred = pipe_all.predict(X_test)
Y_val_pred  = pipe_all.predict(X_val)

all_test_metrics = multi_metrics(Y_test.values, Y_test_pred)
all_val_metrics  = multi_metrics(Y_val.values,  Y_val_pred)

mt_all = pipe_all.named_steps["model"]
coef_all = mt_all.coef_
feat_names = X_train.columns.tolist()
l2 = np.sqrt((coef_all ** 2).sum(axis=0))
imp_df = (pd.DataFrame({"feature": feat_names, "l2_norm_across_tasks": l2})
            .sort_values("l2_norm_across_tasks", ascending=False)
            .reset_index(drop=True))
imp_df.to_csv(outdir / "importances/multitask_enet_importance_l2_train.csv", index=False)

# top 10 retrain
top_features = imp_df.head(min(TOP_K, len(imp_df)))["feature"].tolist()
pd.Series(top_features, name="feature").to_csv(outdir / "top_features/multitask_enet_selected_top10.csv", index=False)

selector = ColumnTransformer([("sel", "passthrough", top_features)], remainder="drop")

pipe_top = Pipeline([
    ("select", selector),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", MultiTaskElasticNetCV(
        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        alphas=np.logspace(-3, 3, 60),  # stronger & wider regularization grid
        cv=inner_cv,
        random_state=seed,
        max_iter=200000,  # more iterations
        tol=1e-3,  # slightly looser tolerance
        n_jobs=-1,
    ))
])

# CV on TRAIN (5-fold mean) for the top-10
cv_top = cv_means(pipe_top, X_train, Y_train, n_splits=5)

# Fit on TRAIN and evaluate on TEST + VAL
pipe_top.fit(X_train, Y_train)
Y_test_pred_top = pipe_top.predict(X_test)
Y_val_pred_top  = pipe_top.predict(X_val)

top_test_metrics = multi_metrics(Y_test.values, Y_test_pred_top)
top_val_metrics  = multi_metrics(Y_val.values,  Y_val_pred_top)

rows = [
    {
        "model": "multitask_enet_all",
        "n_features": len(feat_names),
        **cv_all,
        "test_r2": all_test_metrics["r2"], "test_mae": all_test_metrics["mae"], "test_rmse": all_test_metrics["rmse"],
        "val_r2":  all_val_metrics["r2"],  "val_mae":  all_val_metrics["mae"],  "val_rmse":  all_val_metrics["rmse"],
    },
    {
        "model": "multitask_enet_top10",
        "n_features": len(top_features),
        **cv_top,
        "test_r2": top_test_metrics["r2"], "test_mae": top_test_metrics["mae"], "test_rmse": top_test_metrics["rmse"],
        "val_r2":  top_val_metrics["r2"],  "val_mae":  top_val_metrics["mae"],  "val_rmse":  top_val_metrics["rmse"],
    },
]
pd.DataFrame(rows).to_csv(outdir / "performances/metrics_trainCV_test_val.csv", index=False)

dump({"pipeline": pipe_all,
      "ligand_features": feat_names,
      "tasks": Y_all.columns.tolist()},
     outdir / "models/multitask_enet_all.joblib")

dump({"pipeline": pipe_top,
      "ligand_features": top_features,
      "tasks": Y_all.columns.tolist()},
     outdir / "models/multitask_enet_top10.joblib")

print("\n[OK] Saved:")
print(" - importances:", outdir / "importances/multitask_enet_importance_l2_train.csv")
print(" - top10 list :", outdir / "top_features/multitask_enet_selected_top10.csv")
print(" - metrics    :", outdir / "performances/metrics_trainCV_test_val.csv")
print(" - models     :", outdir / "models/multitask_enet_all.joblib",
      "and", outdir / "models/multitask_enet_top10.joblib")
print(f"Tasks (substrates): {Y_all.columns.tolist()}")
print(f"Top-10 ligand features (joint across tasks): {top_features}")
