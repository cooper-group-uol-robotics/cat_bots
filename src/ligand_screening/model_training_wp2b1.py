import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump
import shap

seed = 42
np.random.seed(seed)

def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": r2_score(y_true, y_pred), "mae": mean_absolute_error(y_true, y_pred), "rmse": rmse}

def fit_transform_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame, X_unseen: pd.DataFrame):
    feature_names = X_train.columns.tolist()
    preproc = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    Xtr = preproc.fit_transform(X_train)
    Xte = preproc.transform(X_test)
    Xun = preproc.transform(X_unseen)
    return Xtr, Xte, Xun, feature_names, preproc

def shap_rank_blackbox(predict_fn, X_bg, X_eval, feature_names):
    expl = shap.KernelExplainer(predict_fn, X_bg)
    sv = expl.shap_values(X_eval, nsamples="auto", l1_reg="num_features(10)")
    mean_abs = np.abs(sv).mean(axis=0)
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)

def subset_by_features(X, feature_names, selected_features):
    name_to_idx = {f: i for i, f in enumerate(feature_names)}
    idx = [name_to_idx[f] for f in selected_features if f in name_to_idx]
    return X[:, idx], idx

def cv_means_from_pipeline(pipe, X_df, y_series, cv=5):
    scorers = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }
    cvres = cross_validate(pipe, X_df, y_series, cv=cv, scoring=scorers, n_jobs=None, return_train_score=False)
    return {
        "cv_r2_mean": float(cvres["test_r2"].mean()),
        "cv_mae_mean": float(-cvres["test_neg_mae"].mean()),
        "cv_rmse_mean": float(-cvres["test_neg_rmse"].mean()),
    }

csv_path = "../data/reactions/wp2a_60_ligand_datapoints.csv"  # descriptors + Yield (%) only
df = pd.read_csv(csv_path)
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].dropna(axis=1, how="all")

target_col = "Yield (%)"
X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
y = df[target_col].astype(float)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=seed)
X_test, X_unseen, y_test, y_unseen = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed)


Xtr, Xte, Xun, feature_names, preproc = fit_transform_numeric(X_train, X_test, X_unseen)

# ElasticNet (with internal CV for hyperparams)
enet = ElasticNetCV(
    alphas=np.logspace(-4, 2, 30),
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    cv=5, max_iter=10000, random_state=seed
).fit(Xtr, y_train)


coef = enet.coef_.ravel()
enet_importance = pd.DataFrame({"feature": feature_names, "coef": coef})
enet_importance["abs_coef"] = enet_importance["coef"].abs()
enet_importance = enet_importance.sort_values("abs_coef", ascending=False).reset_index(drop=True)

# MLP
mlp = MLPRegressor(
    hidden_layer_sizes=(92, 92, 92),
    activation="tanh",
    solver="adam",
    alpha=1e-2,
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=seed
).fit(Xtr, y_train)


# GPR
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(Xtr.shape[1]), length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True,
                               n_restarts_optimizer=3, random_state=seed).fit(Xtr, y_train)

pipe_enet_all = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler()),
                          ("model", ElasticNetCV(alphas=np.logspace(-4,2,30),
                                                 l1_ratio=[0.1,0.3,0.5,0.7,0.9,1.0],
                                                 cv=5, max_iter=10000, random_state=seed))])
cv_enet_all = cv_means_from_pipeline(pipe_enet_all, X_train, y_train, cv=5)

pipe_mlp_all = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("model", MLPRegressor(hidden_layer_sizes=(64,), activation="relu",
                                                alpha=1e-4, learning_rate_init=1e-3,
                                                max_iter=2000, early_stopping=True, n_iter_no_change=20,
                                                random_state=seed))])
cv_mlp_all = cv_means_from_pipeline(pipe_mlp_all, X_train, y_train, cv=5)

pipe_gpr_all = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("model", GaussianProcessRegressor(
                             kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2,1e3))
                                    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5,1e2)),
                             alpha=0.0, normalize_y=True, n_restarts_optimizer=1, random_state=seed))])
cv_gpr_all = cv_means_from_pipeline(pipe_gpr_all, X_train, y_train, cv=5)


enet_test   = regression_metrics(y_test,   enet.predict(Xte))
enet_unseen = regression_metrics(y_unseen, enet.predict(Xun))

mlp_test    = regression_metrics(y_test,   mlp.predict(Xte))
mlp_unseen  = regression_metrics(y_unseen, mlp.predict(Xun))

gpr_test    = regression_metrics(y_test,   gpr.predict(Xte))
gpr_unseen  = regression_metrics(y_unseen, gpr.predict(Xun))


TOP_K = min(10, len(feature_names))

# ElasticNet top-10 (filter true non-zeros first)
tol = 1e-12
enet_nz = enet_importance[enet_importance["abs_coef"] > tol]
k_enet = min(TOP_K, len(enet_nz))
enet_top10 = enet_nz.head(k_enet)["feature"].tolist()

# MLP & GPR SHAP rankings on TRAIN ONLY (no peeking at test/unseen)
rng = np.random.default_rng(seed)
bg_n = min(50, Xtr.shape[0]); ev_n = min(50, Xtr.shape[0])
bg_idx = rng.choice(Xtr.shape[0], size=bg_n, replace=False)
ev_idx = rng.choice(Xtr.shape[0], size=ev_n, replace=False)

mlp_shap = shap_rank_blackbox(mlp.predict, Xtr[bg_idx], Xtr[ev_idx], feature_names)
gpr_shap = shap_rank_blackbox(gpr.predict, Xtr[bg_idx], Xtr[ev_idx], feature_names)

mlp_top10 = mlp_shap.head(TOP_K)["feature"].tolist()
gpr_top10 = gpr_shap.head(TOP_K)["feature"].tolist()

# Subset matrices
Xtr_enet10, idx_enet10 = subset_by_features(Xtr, feature_names, enet_top10)
Xte_enet10, _ = subset_by_features(Xte, feature_names, enet_top10)
Xun_enet10, _ = subset_by_features(Xun, feature_names, enet_top10)

Xtr_mlp10, idx_mlp10 = subset_by_features(Xtr, feature_names, mlp_top10)
Xte_mlp10, _ = subset_by_features(Xte, feature_names, mlp_top10)
Xun_mlp10, _ = subset_by_features(Xun, feature_names, mlp_top10)

Xtr_gpr10, idx_gpr10 = subset_by_features(Xtr, feature_names, gpr_top10)
Xte_gpr10, _ = subset_by_features(Xte, feature_names, gpr_top10)
Xun_gpr10, _ = subset_by_features(Xun, feature_names, gpr_top10)

# retrain on top10
enet10 = ElasticNetCV(alphas=np.logspace(-4, 2, 30),
                      l1_ratio=[0.1,0.3,0.5,0.7,0.9,1.0],
                      cv=5, max_iter=10000, random_state=seed).fit(Xtr_enet10, y_train)
mlp10 = MLPRegressor(hidden_layer_sizes=(64,), activation="relu",
                     alpha=1e-4, learning_rate_init=1e-3,
                     max_iter=2000, early_stopping=True, n_iter_no_change=20,
                     random_state=seed).fit(Xtr_mlp10, y_train)
gpr10 = GaussianProcessRegressor(
    kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(Xtr_gpr10.shape[1]), length_scale_bounds=(1e-2, 1e3))
           + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2)),
    alpha=0.0, normalize_y=True, n_restarts_optimizer=3, random_state=seed
).fit(Xtr_gpr10, y_train)

# CV means for the TOP-10 (selection fixed from train set)
def make_topk_pipe(selected_names, estimator):
    selector = ColumnTransformer([("sel", "passthrough", selected_names)], remainder="drop")
    return Pipeline([("select", selector),
                     ("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("model", estimator)])

pipe_enet10 = make_topk_pipe(enet_top10, ElasticNetCV(alphas=np.logspace(-4,2,30),
                                                      l1_ratio=[0.1,0.3,0.5,0.7,0.9,1.0],
                                                      cv=5, max_iter=10000, random_state=seed))
pipe_mlp10  = make_topk_pipe(mlp_top10, MLPRegressor(hidden_layer_sizes=(64,), activation="relu",
                                                      alpha=1e-4, learning_rate_init=1e-3,
                                                      max_iter=2000, early_stopping=True, n_iter_no_change=20,
                                                      random_state=seed))
pipe_gpr10  = make_topk_pipe(gpr_top10, GaussianProcessRegressor(
                                kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2,1e3))
                                       + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5,1e2)),
                                alpha=0.0, normalize_y=True, n_restarts_optimizer=1, random_state=seed))

cv_enet10 = cv_means_from_pipeline(pipe_enet10, X_train, y_train, cv=5)
cv_mlp10  = cv_means_from_pipeline(pipe_mlp10,  X_train, y_train, cv=5)
cv_gpr10  = cv_means_from_pipeline(pipe_gpr10,  X_train, y_train, cv=5)

enet10_test   = regression_metrics(y_test, enet10.predict(Xte_enet10))
enet10_unseen = regression_metrics(y_unseen, enet10.predict(Xun_enet10))

mlp10_test    = regression_metrics(y_test, mlp10.predict(Xte_mlp10))
mlp10_unseen  = regression_metrics(y_unseen, mlp10.predict(Xun_mlp10))

gpr10_test    = regression_metrics(y_test, gpr10.predict(Xte_gpr10))
gpr10_unseen  = regression_metrics(y_unseen, gpr10.predict(Xun_gpr10))

outdir = Path("../data/ML/first_training")
(outdir / "importances").mkdir(parents=True, exist_ok=True)
(outdir / "top_features").mkdir(parents=True, exist_ok=True)
(outdir / "models").mkdir(parents=True, exist_ok=True)
(outdir / "performances").mkdir(parents=True, exist_ok=True)

# importances / rankings
enet_importance.to_csv(outdir / "importances/elasticnet_coef_importance.csv", index=False)
mlp_shap.to_csv(outdir / "importances/mlp_shap_importance.csv", index=False)
gpr_shap.to_csv(outdir / "importances/gpr_shap_importance.csv", index=False)

# top-10 feature lists
pd.Series(enet_top10, name="feature").to_csv(outdir / "top_features/elasticnet_selected_top10.csv", index=False)
pd.Series(mlp_top10,  name="feature").to_csv(outdir / "top_features/mlp_selected_top10.csv", index=False)
pd.Series(gpr_top10,  name="feature").to_csv(outdir / "top_features/gpr_selected_top10.csv", index=False)

# preprocessor & models (all + top10)
dump(preproc, outdir / "preproc.joblib")

all_idx = list(range(len(feature_names)))
dump({"model": enet, "selected_features": feature_names, "feature_indices": all_idx}, outdir / "models/elasticnet_all.joblib")
dump({"model": mlp,  "selected_features": feature_names, "feature_indices": all_idx}, outdir / "models/mlp_all.joblib")
dump({"model": gpr,  "selected_features": feature_names, "feature_indices": all_idx}, outdir / "models/gpr_all.joblib")

dump({"model": enet10, "selected_features": enet_top10, "feature_indices": idx_enet10}, outdir / "models/elasticnet_top10.joblib")
dump({"model": mlp10,  "selected_features": mlp_top10,  "feature_indices": idx_mlp10},  outdir / "models/mlp_top10.joblib")
dump({"model": gpr10,  "selected_features": gpr_top10,  "feature_indices": idx_gpr10},  outdir / "models/gpr_top10.joblib")

rows_all = [
    {"model": "ElasticNet",
     **cv_enet_all,
     "test_r2": enet_test["r2"], "test_mae": enet_test["mae"], "test_rmse": enet_test["rmse"],
     "unseen_r2": enet_unseen["r2"], "unseen_mae": enet_unseen["mae"], "unseen_rmse": enet_unseen["rmse"],
     "n_features": len(feature_names),
     "alpha": float(enet.alpha_), "l1_ratio": float(enet.l1_ratio_) if hasattr(enet, "l1_ratio_") else np.nan},
    {"model": "MLP",
     **cv_mlp_all,
     "test_r2": mlp_test["r2"], "test_mae": mlp_test["mae"], "test_rmse": mlp_test["rmse"],
     "unseen_r2": mlp_unseen["r2"], "unseen_mae": mlp_unseen["mae"], "unseen_rmse": mlp_unseen["rmse"],
     "n_features": len(feature_names),
     "hidden": str(mlp.hidden_layer_sizes)},
    {"model": "GPR",
     **cv_gpr_all,
     "test_r2": gpr_test["r2"], "test_mae": gpr_test["mae"], "test_rmse": gpr_test["rmse"],
     "unseen_r2": gpr_unseen["r2"], "unseen_mae": gpr_unseen["mae"], "unseen_rmse": gpr_unseen["rmse"],
     "n_features": len(feature_names),
     "kernel": str(gpr.kernel_)},
]
pd.DataFrame(rows_all).to_csv(outdir / "performances/metrics_all_features.csv", index=False)

rows_top = [
    {"model": "ElasticNet_top10",
     **cv_enet10,
     "test_r2": enet10_test["r2"], "test_mae": enet10_test["mae"], "test_rmse": enet10_test["rmse"],
     "unseen_r2": enet10_unseen["r2"], "unseen_mae": enet10_unseen["mae"], "unseen_rmse": enet10_unseen["rmse"],
     "n_features": len(enet_top10)},
    {"model": "MLP_top10",
     **cv_mlp10,
     "test_r2": mlp10_test["r2"], "test_mae": mlp10_test["mae"], "test_rmse": mlp10_test["rmse"],
     "unseen_r2": mlp10_unseen["r2"], "unseen_mae": mlp10_unseen["mae"], "unseen_rmse": mlp10_unseen["rmse"],
     "n_features": len(mlp_top10)},
    {"model": "GPR_top10",
     **cv_gpr10,
     "test_r2": gpr10_test["r2"], "test_mae": gpr10_test["mae"], "test_rmse": gpr10_test["rmse"],
     "unseen_r2": gpr10_unseen["r2"], "unseen_mae": gpr10_unseen["mae"], "unseen_rmse": gpr10_unseen["rmse"],
     "n_features": len(gpr_top10)},
]
pd.DataFrame(rows_top).to_csv(outdir / "performances/metrics_top10.csv", index=False)

print("\nSaved metrics to:",
      outdir / "performances/metrics_all_features.csv",
      "and",
      outdir / "performances/metrics_top10.csv")
