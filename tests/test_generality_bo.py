import numpy as np
import pandas as pd

from catbots.bo import fit_gp, group_mean_posterior
from catbots.campaign import run_generality_bo
from catbots.oracle import ReactionOracle


def test_group_mean_posterior_matches_full_covariance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    gp = fit_gp(X[:15], 100 * rng.random(15), random_state=0)
    groups = [np.arange(15, 25), np.arange(25, 40), np.array([3, 30, 31])]
    mu, sd, _ = group_mean_posterior(gp, X, groups)
    for g, m, s in zip(groups, mu, sd):
        mean, cov = gp.predict(X[g], return_cov=True)
        assert np.isclose(m, mean.mean())
        assert np.isclose(s, np.sqrt(cov.sum()) / len(g), rtol=1e-4)


def test_generality_bo_spends_exact_budget_without_repeats():
    grid = pd.DataFrame([(l, s) for l in "abcdef" for s in range(8)], columns=["ligand", "substrate"])
    grid["yield"] = np.random.default_rng(1).random(len(grid)) * 100
    o = ReactionOracle("toy", grid, factors=("ligand", "substrate"))
    X = pd.get_dummies(grid[["ligand", "substrate"]].astype(str), dtype=float).to_numpy()
    r = run_generality_bo(o, X, 20, seed=0, condition=["ligand"])
    assert len(r.rows) == 20 == len(set(r.rows)) == o.n_queries
