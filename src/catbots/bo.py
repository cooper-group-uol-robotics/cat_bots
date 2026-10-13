"""Gaussian-process Bayesian optimisation over a discrete, enumerated candidate set."""

import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sd = np.maximum(sd, 1e-12)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def fit_gp(X: np.ndarray, y: np.ndarray, random_state: int | None = None,
           kernel=None) -> GaussianProcessRegressor:
    """ARD Matern-5/2 + white-noise GP on yields scaled to 0-1 (predictions are on that scale).

    Passing a fitted ``kernel`` (``gp.kernel_``) reuses its hyperparameters without re-optimising them.
    """
    if kernel is None:
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(1e-2, (1e-6, 1.0))
        )
        optimizer = "fmin_l_bfgs_b"
    else:
        optimizer = None
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=optimizer,
                                  random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, np.asarray(y, dtype=float) / 100.0)
    return gp


def group_mean_posterior(gp: GaussianProcessRegressor, X: np.ndarray, groups: list[np.ndarray],
                         K_sums: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean and SD of the *average* of f over each group of rows of ``X`` (on the 0-1 scale).

    Uses the full within-group posterior covariance: var(mean_g) = (1'K_gg 1 - |V_g 1|^2) / n_g^2 with
    V = L^-1 K(X_train, X). ``K_sums`` (1'K_gg 1 per group) depends only on the kernel and can be reused.
    """
    from scipy.linalg import solve_triangular

    K_star = gp.kernel_(gp.X_train_, X)
    mu = (K_star.T @ gp.alpha_) * gp._y_train_std + gp._y_train_mean
    V = solve_triangular(gp.L_, K_star, lower=True)
    if K_sums is None:
        K_sums = np.array([gp.kernel_(X[g]).sum() for g in groups])
    n = np.array([len(g) for g in groups], dtype=float)
    Vsum = np.stack([V[:, g].sum(axis=1) for g in groups], axis=1)
    var = np.maximum(K_sums - (Vsum ** 2).sum(axis=0), 1e-12) / n ** 2 * gp._y_train_std ** 2
    means = np.array([mu[g].mean() for g in groups])
    return means, np.sqrt(var), K_sums


class GPExpectedImprovement:
    """GP surrogate (ARD Matern-5/2 + white noise) with expected improvement, batch size 1.

    Kernel hyperparameters are re-optimised every ``refit_every`` suggestions and reused (exact GP
    update, no optimisation) in between; ``refit_every=1`` re-optimises at every step.
    """

    def __init__(self, xi: float = 0.01, random_state: int | None = None, refit_every: int = 5):
        self.xi = xi
        self.random_state = random_state
        self.refit_every = refit_every
        self._kernel = None
        self._calls = 0

    def suggest(self, X_train: np.ndarray, y_train: np.ndarray, X_candidates: np.ndarray) -> int:
        """Index (into ``X_candidates``) of the candidate with the highest expected improvement."""
        reuse = self._kernel is not None and self._calls % self.refit_every != 0
        gp = fit_gp(X_train, y_train, self.random_state, kernel=self._kernel if reuse else None)
        self._kernel = gp.kernel_
        self._calls += 1
        mu, sd = gp.predict(X_candidates, return_std=True)
        return int(np.argmax(expected_improvement(mu, sd, np.max(y_train) / 100.0, self.xi)))
