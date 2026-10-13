"""Low-data models of the WP2 pipeline (ported from model_training_wp2b1.py, model_training_wp2c.py and
multi_task_model_training_wp2c_loo.py).

Differences from the scripts, forced by the data sizes the overview prescribes (12-72 points):
model choice uses leave-one-out CV instead of 80/10/10 splits, which leave 1-3 test points.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV, MultiTaskElasticNetCV
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return fn(*args, **kwargs)


def _regressors(n_samples: int, n_features: int, seed: int) -> dict:
    cv = min(5, n_samples)
    return {
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNetCV(
            alphas=np.logspace(-4, 2, 30), l1_ratio=L1_RATIOS, cv=cv, max_iter=10000, random_state=seed)),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64,), alpha=1e-4, max_iter=2000, random_state=seed)),
        "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(n_features), (1e-2, 1e3))
            + WhiteKernel(1.0, (1e-5, 1e2)), normalize_y=True, random_state=seed)),
    }


@dataclass
class RegressorChoice:
    name: str
    model: object               # fitted on all data
    loo_r2: dict                # LOO-CV R^2 of every candidate model
    feature_ranking: pd.Series  # importance per descriptor, descending


def _importance(name, model, X: pd.DataFrame, seed: int) -> pd.Series:
    if name == "ElasticNet":
        imp = np.abs(model[-1].coef_)
    else:  # SHAP for the non-linear models, as in the overview
        import shap
        np.random.seed(seed)  # KernelExplainer samples coalitions with the global numpy RNG
        # wrap in a plain function: shap tries to overwrite Pipeline.feature_names_in_ on bound methods
        expl = shap.KernelExplainer(lambda Z: model.predict(pd.DataFrame(Z, columns=X.columns)), X.to_numpy())
        imp = np.abs(_quiet(expl.shap_values, X.to_numpy(), nsamples=200, silent=True)).mean(axis=0)
    return pd.Series(imp, index=X.columns).sort_values(ascending=False)


def select_regressor(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> RegressorChoice:
    """WP2B: fit ElasticNet, shallow NN and GPR; pick the best by LOO-CV R^2 and rank its features."""
    y = np.asarray(y, dtype=float)
    cands = _regressors(len(X), X.shape[1], seed)
    r2 = {}
    for name, m in cands.items():
        pred = _quiet(cross_val_predict, m, X, y, cv=LeaveOneOut())
        r2[name] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    best = max(r2, key=r2.get)
    model = _quiet(cands[best].fit, X, y)
    return RegressorChoice(best, model, r2, _importance(best, model, X, seed))


@dataclass
class ActivityClassifier:
    model: object | None        # None when the training labels have a single class
    features: list[str]         # top features by |coef|
    constant_label: int | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self.constant_label, dtype=int)
        return self.model.predict(X).astype(int)


def fit_activity_classifier(X: pd.DataFrame, active: np.ndarray, n_features: int = 5,
                            seed: int = 0) -> ActivityClassifier:
    """WP2C: ElasticNet logistic regression for active (>= threshold) vs inactive substrates."""
    active = np.asarray(active, dtype=int)
    counts = np.bincount(active, minlength=2)
    if counts.min() == 0:
        return ActivityClassifier(None, [], constant_label=int(active[0]))
    folds = int(min(5, counts.min()))
    if folds >= 2:
        clf = LogisticRegressionCV(Cs=np.logspace(-4, 4, 20), l1_ratios=L1_RATIOS, solver="saga",
                                   cv=StratifiedKFold(folds, shuffle=True, random_state=seed), scoring="roc_auc",
                                   class_weight="balanced", max_iter=5000, random_state=seed,
                                   use_legacy_attributes=False)
    else:
        clf = LogisticRegression(l1_ratio=0.5, C=1.0, solver="saga", class_weight="balanced",
                                 max_iter=5000, random_state=seed)
    model = _quiet(make_pipeline(StandardScaler(), clf).fit, X, active)
    coef = pd.Series(np.abs(model[-1].coef_.ravel()), index=X.columns).sort_values(ascending=False)
    top = coef[coef > 1e-12].index[:n_features].tolist() or coef.index[:n_features].tolist()
    return ActivityClassifier(model, top)


@dataclass
class MultitaskChoice:
    model: object
    tasks: list
    feature_ranking: pd.Series  # L2 norm of coefficients across tasks
    y_mean: np.ndarray
    y_std: np.ndarray

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Y = self.model.predict(X) * self.y_std + self.y_mean
        return pd.DataFrame(Y, index=X.index, columns=self.tasks)


def fit_multitask(X: pd.DataFrame, Y: pd.DataFrame, seed: int = 0) -> MultitaskChoice:
    """WP2C: multitask ElasticNet (one task per substrate) on ligand descriptors, per-task standardised Y."""
    y_mean, y_std = Y.mean().to_numpy(), Y.std().replace(0, 1.0).to_numpy()
    model = make_pipeline(StandardScaler(), MultiTaskElasticNetCV(
        l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95], alphas=np.logspace(-3, 3, 60),
        cv=KFold(min(5, len(X)), shuffle=True, random_state=seed), max_iter=200000, tol=1e-3, random_state=seed))
    _quiet(model.fit, X, (Y.to_numpy() - y_mean) / y_std)
    l2 = np.sqrt((model[-1].coef_ ** 2).sum(axis=0))
    return MultitaskChoice(model, Y.columns.tolist(), pd.Series(l2, index=X.columns).sort_values(ascending=False),
                           y_mean, y_std)
