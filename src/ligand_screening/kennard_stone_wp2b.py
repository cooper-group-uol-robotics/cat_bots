import math
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ligands_60  = pd.read_csv("../data/ligands/ks60.csv")
ligands_348 = pd.read_csv("../data/ligands/fey_monop.csv")

ligands_60["SMILES"]  = ligands_60["SMILES"].astype(str).str.strip()
ligands_348["SMILES"] = ligands_348["SMILES"].astype(str).str.strip()

ligands_288 = ligands_348[~ligands_348["SMILES"].isin(ligands_60["SMILES"])].copy()
ligands_288.reset_index(drop=True, inplace=True)

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.match(r"^Unnamed")].dropna(axis=1, how="all")

def _align_and_transform(df: pd.DataFrame, selected_features: list[str], preproc):

    df = _clean_columns(df).copy()

    try:
        train_features = list(preproc.named_steps["imputer"].feature_names_in_)
    except Exception:

        train_features = selected_features

    X_full_df = df.filter(items=train_features).reindex(columns=train_features)
    X_full_df = X_full_df.apply(pd.to_numeric, errors="coerce")
    Xtr_full = preproc.transform(X_full_df)

    X_sel_df = df.filter(items=selected_features).reindex(columns=selected_features)
    X_sel_df = X_sel_df.apply(pd.to_numeric, errors="coerce")

    return Xtr_full, X_sel_df

def predict_with_bundle(df_new: pd.DataFrame, preproc_path: str, bundle_path: str, return_std: bool = False):
    preproc = load(preproc_path)
    bundle = load(bundle_path)

    model = bundle["model"]
    selected_features = bundle["selected_features"]
    idx = bundle["feature_indices"]

    Xtr_full, df_selected_only = _align_and_transform(df_new, selected_features, preproc)
    Xsub = Xtr_full[:, idx]

    if return_std:
        try:
            y_pred, y_std = model.predict(Xsub, return_std=True)
            return df_selected_only, np.asarray(y_pred).ravel(), np.asarray(y_std).ravel(), selected_features, idx
        except TypeError:
            pass

    y_pred = model.predict(Xsub)
    return df_selected_only, np.asarray(y_pred).ravel(), None, selected_features, idx

def kennard_stone_indices(X: np.ndarray, k: int) -> list[int]:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n_samples ({n})")
    G = X @ X.T
    H = np.sum(X**2, axis=1, keepdims=True)
    D2 = H + H.T - 2 * G
    np.fill_diagonal(D2, 0.0)
    i1, i2 = np.unravel_index(np.argmax(D2), D2.shape)
    selected = [int(i1), int(i2)]
    min_d2 = np.minimum(D2[:, i1], D2[:, i2])
    min_d2[i1] = -np.inf; min_d2[i2] = -np.inf
    while len(selected) < k:
        j = int(np.argmax(min_d2))
        selected.append(j)
        min_d2 = np.minimum(min_d2, D2[:, j])
        min_d2[j] = -np.inf
    return selected


metrics = pd.read_csv("../data/ML/performances/metrics_top10.csv")
TEST_W, CV_W = 0.6, 0.4
metrics["overall_score"] = TEST_W*metrics["test_r2"] + CV_W*metrics["cv_r2_mean"]
best = metrics.sort_values(["overall_score", "unseen_r2"], ascending=[False, False]).iloc[0]
best_name = best["model"]

bundle_map = {
    "ElasticNet_top10": "../data/ML/models/elasticnet_top10.joblib",
    "MLP_top10":        "../data/ML/models/mlp_top10.joblib",
    "GPR_top10":        "../data/ML/models/gpr_top10.joblib",
}
bundle_path = bundle_map[best_name]
preproc_path = "../data/ML/preproc.joblib"

use_std = (best_name == "GPR_top10")
df_sel, y_hat, y_std, sel_feats, sel_idx = predict_with_bundle(
    ligands_288, preproc_path, bundle_path, return_std=use_std
)


base_cols = [c for c in ["No.", "SMILES"] if c in ligands_288.columns]
preds = ligands_288[base_cols].copy() if base_cols else pd.DataFrame(index=ligands_288.index)
preds = preds.join(df_sel)
preds["predicted_yield"] = y_hat
preds["model_used"] = best_name

if use_std:
    preds["predicted_yield_std"] = y_std


k_top = max(1, math.ceil(0.15 * len(preds)))
top15 = preds.nlargest(k_top, "predicted_yield").copy()


X_top = top15[sel_feats].to_numpy()
X_top = SimpleImputer(strategy="median").fit_transform(X_top)
X_top = StandardScaler().fit_transform(X_top)

k_ks = min(12, X_top.shape[0])
order = kennard_stone_indices(X_top, k_ks)

ks12 = top15.iloc[order].copy()
ks12.insert(0, "ks_rank", range(1, k_ks + 1))

top15["predicted_yield"] = np.rint(top15["predicted_yield"]).astype(int)
ks12["predicted_yield"]  = np.rint(ks12["predicted_yield"]).astype(int)


if "predicted_yield_std" in top15:
    top15["predicted_yield_std"] = top15["predicted_yield_std"].round(1)
if "predicted_yield_std" in ks12:
    ks12["predicted_yield_std"] = ks12["predicted_yield_std"].round(1)

outdir = Path("../data/ligands")
outdir.mkdir(parents=True, exist_ok=True)
top15.to_csv(outdir / "predicted_ligands_top15pct.csv", index=False)
ks12.to_csv(outdir / "predicted_ligands_ks12.csv", index=False)

print(f"Best model: {best_name} | overall={best['overall_score']:.3f} | "
      f"test_r2={best['test_r2']:.3f} | cv_r2_mean={best['cv_r2_mean']:.3f}")
print("Saved:", outdir / "predicted_ligands_top15pct.csv")
print("Saved:", outdir / "predicted_ligands_ks12.csv")