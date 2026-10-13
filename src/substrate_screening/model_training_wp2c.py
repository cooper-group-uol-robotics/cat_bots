# train_logreg_elasticnet_auc_acc_mcc_stratified_top5.py
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, make_scorer
from joblib import dump

seed = 42
np.random.seed(seed)

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float("nan")  # if a fold/split has a single class

def cls_metrics(y_true, y_proba, thresh=0.5):
    y_pred = (y_proba >= thresh).astype(int)
    return {
        "auc": safe_auc(y_true, y_proba),
        "acc": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

def fit_transform_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame, X_unseen: pd.DataFrame):
    feature_names = X_train.columns.tolist()
    preproc = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    Xtr = preproc.fit_transform(X_train)
    Xte = preproc.transform(X_test)
    Xun = preproc.transform(X_unseen)
    return Xtr, Xte, Xun, feature_names, preproc

def subset_by_features(X, feature_names, selected_features):
    name_to_idx = {f: i for i, f in enumerate(feature_names)}
    idx = [name_to_idx[f] for f in selected_features if f in name_to_idx]
    return X[:, idx], idx

def cv_means_from_pipeline(pipe, X_df, y_series, cv):
    if cv is None:
        return {"cv_auc_mean": np.nan, "cv_acc_mean": np.nan, "cv_mcc_mean": np.nan}
    scorers = {
        "auc": "roc_auc",
        "acc": "accuracy",
        "mcc": make_scorer(matthews_corrcoef),
    }
    cvres = cross_validate(pipe, X_df, y_series, cv=cv, scoring=scorers,
                           n_jobs=None, return_train_score=False)
    return {
        "cv_auc_mean": float(cvres["test_auc"].mean()),
        "cv_acc_mean": float(cvres["test_acc"].mean()),
        "cv_mcc_mean": float(cvres["test_mcc"].mean()),
    }

def make_stratified_kfold_if_possible(y, desired=5):
    vc = pd.Series(y).value_counts()
    if len(vc) < 2:
        return None
    n_splits = int(min(desired, vc.min()))
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def try_stratified_second_split(X_temp, y_temp):
    vc = y_temp.value_counts()
    if len(vc) == 2 and vc.min() >= 2:
        return train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    return train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=None)

def make_topk_pipe(selected_names, estimator):
    selector = ColumnTransformer([("sel", "passthrough", selected_names)], remainder="drop")
    return Pipeline([("select", selector),
                     ("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("model", estimator)])

def get_param_or_nan(model, attr, fallback=None):
    if hasattr(model, attr):
        v = getattr(model, attr)
        try:
            return float(np.atleast_1d(v).mean())
        except Exception:
            return np.nan
    return float(fallback) if fallback is not None else np.nan

# this dataset should contain descriptors, SMILES and "Yield (%)"
csv_path = "../data/reactions/wp2c_36_substrate_datapoints.csv"
df = pd.read_csv(csv_path)

# clean
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].dropna(axis=1, how="all")

target_col = "Yield (%)"
id_cols = [c for c in ["No.", "SMILES"] if c in df.columns]

X = df.drop(columns=[target_col] + id_cols, errors="ignore").select_dtypes(include=[np.number])
if X.shape[1] == 0:
    raise ValueError("No numeric descriptor columns found after removing target and ID columns (No., SMILES).")

y_reg = df[target_col].astype(float)
y = (y_reg >= 5.0).astype(int)

# 80/10/10 stratified if possible
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
X_test, X_unseen, y_test, y_unseen = try_stratified_second_split(X_temp, y_temp)

Xtr, Xte, Xun, feature_names, preproc = fit_transform_numeric(X_train, X_test, X_unseen)

cv_train = make_stratified_kfold_if_possible(y_train, desired=5)

L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
C_GRID  = np.logspace(-4, 4, 20)

if cv_train is not None:
    log_all = LogisticRegressionCV(
        Cs=C_GRID, cv=cv_train, penalty="elasticnet", solver="saga",
        l1_ratios=L1_GRID, scoring="roc_auc", class_weight="balanced",
        max_iter=5000, n_jobs=None, random_state=seed, refit=True
    ).fit(Xtr, y_train)
else:
    log_all = LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=0.5, C=1.0, class_weight="balanced",
        max_iter=5000, random_state=seed
    ).fit(Xtr, y_train)

# importances (on standardized space)
coef = log_all.coef_.ravel()
imp_df = pd.DataFrame({"feature": feature_names, "coef": coef})
imp_df["abs_coef"] = imp_df["coef"].abs()
imp_df = imp_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

# CV means on TRAIN via pipeline (no leakage)
pipe_all = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegressionCV(
        Cs=C_GRID, cv=cv_train if cv_train is not None else 2,
        penalty="elasticnet", solver="saga",
        l1_ratios=L1_GRID, scoring="roc_auc", class_weight="balanced",
        max_iter=5000, n_jobs=None, random_state=seed, refit=True
    ))
])
cv_all = cv_means_from_pipeline(pipe_all, X_train, y_train, cv=cv_train)

# Test & unseen metrics
proba_te = log_all.predict_proba(Xte)[:, 1]
proba_un = log_all.predict_proba(Xun)[:, 1]
test_all   = cls_metrics(y_test,   proba_te, thresh=0.5)
unseen_all = cls_metrics(y_unseen, proba_un, thresh=0.5)

# TOP 5
TOP_K = min(5, len(feature_names))
tol = 1e-12
imp_nz = imp_df[imp_df["abs_coef"] > tol]
k_top = min(TOP_K, len(imp_nz))
top5 = imp_nz.head(k_top)["feature"].tolist()

Xtr_5, idx_5 = subset_by_features(Xtr, feature_names, top5)
Xte_5, _     = subset_by_features(Xte, feature_names, top5)
Xun_5, _     = subset_by_features(Xun, feature_names, top5)

# Retrain on top-5
if cv_train is not None:
    log_5 = LogisticRegressionCV(
        Cs=C_GRID, cv=cv_train, penalty="elasticnet", solver="saga",
        l1_ratios=L1_GRID, scoring="roc_auc", class_weight="balanced",
        max_iter=5000, n_jobs=None, random_state=seed, refit=True
    ).fit(Xtr_5, y_train)
else:
    log_5 = LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=0.5, C=1.0, class_weight="balanced",
        max_iter=5000, random_state=seed
    ).fit(Xtr_5, y_train)

# CV means for top-5 (pipeline with column selector)
pipe_5 = make_topk_pipe(
    top5,
    LogisticRegressionCV(
        Cs=C_GRID, cv=cv_train if cv_train is not None else 2,
        penalty="elasticnet", solver="saga",
        l1_ratios=L1_GRID, scoring="roc_auc", class_weight="balanced",
        max_iter=5000, n_jobs=None, random_state=seed, refit=True
    )
)
cv_5 = cv_means_from_pipeline(pipe_5, X_train, y_train, cv=cv_train)

# Test & unseen on top-5
proba_te_5 = log_5.predict_proba(Xte_5)[:, 1]
proba_un_5 = log_5.predict_proba(Xun_5)[:, 1]
test_5   = cls_metrics(y_test,   proba_te_5, thresh=0.5)
unseen_5 = cls_metrics(y_unseen, proba_un_5, thresh=0.5)

outdir = Path("../data/ML_sub")
(outdir / "importances").mkdir(parents=True, exist_ok=True)
(outdir / "top_features").mkdir(parents=True, exist_ok=True)
(outdir / "models").mkdir(parents=True, exist_ok=True)
(outdir / "performances").mkdir(parents=True, exist_ok=True)

imp_df.to_csv(outdir / "importances/logreg_elasticnet_coef_importance.csv", index=False)
pd.Series(top5, name="feature").to_csv(outdir / "top_features/logreg_selected_top5.csv", index=False)

# save preprocessor & models
dump(preproc, outdir / "preproc.joblib")
all_idx = list(range(len(feature_names)))
dump({"model": log_all, "selected_features": feature_names, "feature_indices": all_idx},
     outdir / "models/logreg_all.joblib")
dump({"model": log_5,  "selected_features": top5, "feature_indices": idx_5},
     outdir / "models/logreg_top5.joblib")

# metrics CSVs
rows_all = [{
    "model": "LogRegEN_all",
    **cv_all,
    "test_auc":   test_all["auc"],   "test_acc":   test_all["acc"],   "test_mcc":   test_all["mcc"],
    "unseen_auc": unseen_all["auc"], "unseen_acc": unseen_all["acc"], "unseen_mcc": unseen_all["mcc"],
    "n_features": len(feature_names),
    "C": get_param_or_nan(log_all, "C_", fallback=getattr(log_all, "C", None)),
    "l1_ratio": get_param_or_nan(log_all, "l1_ratio_", fallback=getattr(log_all, "l1_ratio", None)),
}]
pd.DataFrame(rows_all).to_csv(outdir / "performances/metrics_all_features.csv", index=False)

rows_top = [{
    "model": "LogRegEN_top5",
    **cv_5,
    "test_auc":   test_5["auc"],   "test_acc":   test_5["acc"],   "test_mcc":   test_5["mcc"],
    "unseen_auc": unseen_5["auc"], "unseen_acc": unseen_5["acc"], "unseen_mcc": unseen_5["mcc"],
    "n_features": len(top5),
    "C": get_param_or_nan(log_5, "C_", fallback=getattr(log_5, "C", None)),
    "l1_ratio": get_param_or_nan(log_5, "l1_ratio_", fallback=getattr(log_5, "l1_ratio", None)),
}]
pd.DataFrame(rows_top).to_csv(outdir / "performances/metrics_top5.csv", index=False)

print("Saved metrics to:", outdir / "performances")
