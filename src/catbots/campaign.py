"""Simulated optimisation campaigns against a ReactionOracle, at a fixed experiment budget.

Every strategy spends exactly ``budget`` oracle queries; screening experiments count towards it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from catbots.bo import GPExpectedImprovement, expected_improvement, fit_gp, group_mean_posterior
from catbots.oracle import ReactionOracle
from catbots.screening import screen_factor, top_levels
from catbots.selection import kennard_stone


@dataclass
class CampaignResult:
    benchmark: str
    strategy: str
    seed: int
    rows: list[int]      # oracle.data row indices in the order they were run
    yields: np.ndarray
    phases: list[str]    # "init", "screen" or "bo" per experiment
    screened_factor: str | None = None
    kept_levels: list | None = None

    def best_so_far(self) -> np.ndarray:
        return np.maximum.accumulate(self.yields)


def _query_rows(oracle: ReactionOracle, rows: list[int]) -> np.ndarray:
    return oracle.query(oracle.data.loc[rows, list(oracle.factors)])


def _bo_loop(oracle, X, rows, yields, candidate_mask, n_steps, seed):
    """Extend ``rows``/``yields`` by ``n_steps`` GP-EI suggestions drawn from ``candidate_mask``."""
    bo = GPExpectedImprovement(random_state=seed)
    rows, yields = list(rows), list(yields)
    available = candidate_mask.copy()
    available[rows] = False
    for _ in range(n_steps):
        cand = np.flatnonzero(available)
        if cand.size == 0:
            break
        pick = int(cand[bo.suggest(X[rows], np.asarray(yields), X[cand])])
        rows.append(pick)
        yields.append(float(_query_rows(oracle, [pick])[0]))
        available[pick] = False
    return rows, yields


def run_random(oracle: ReactionOracle, X: np.ndarray, budget: int, seed: int) -> CampaignResult:
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(oracle), size=budget, replace=False).tolist()
    return CampaignResult(oracle.name, "random", seed, rows, _query_rows(oracle, rows), ["random"] * budget)


def run_bo(oracle: ReactionOracle, X: np.ndarray, budget: int, seed: int,
           n_init: int = 10, init: str = "random") -> CampaignResult:
    """BO over the full space; ``init`` is "random" or "kennard_stone" (KS from a random first point)."""
    rng = np.random.default_rng(seed)
    if init == "random":
        rows = rng.choice(len(oracle), size=n_init, replace=False).tolist()
    elif init == "kennard_stone":
        rows = kennard_stone(X, n_init, initial=[int(rng.integers(len(oracle)))])
    else:
        raise ValueError("init must be 'random' or 'kennard_stone'")
    yields = _query_rows(oracle, rows)
    rows, yields = _bo_loop(oracle, X, rows, yields, np.ones(len(oracle), bool), budget - n_init, seed)
    phases = ["init"] * n_init + ["bo"] * (len(rows) - n_init)
    return CampaignResult(oracle.name, f"bo_{init}" if init != "random" else "bo", seed, rows, np.asarray(yields), phases)


def run_generality_bo(oracle: ReactionOracle, X: np.ndarray, budget: int, seed: int,
                      condition: list[str], n_init: int = 10, refit_every: int = 5, xi: float = 0.01) -> CampaignResult:
    """Generality-aware BO: optimises the mean yield over substrates of a condition, not a single yield.

    One GP over (condition, substrate) as in ``run_bo``. Each step picks the **condition** with the highest
    expected improvement of its posterior mean-over-substrates (incumbent: the best predicted mean), then
    measures that condition on its **most uncertain unmeasured substrate**.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(oracle), size=n_init, replace=False).tolist()
    yields = list(_query_rows(oracle, rows))
    codes = oracle.data.groupby(condition, sort=False).ngroup().to_numpy()
    groups = [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]
    measured = np.zeros(len(oracle), bool)
    measured[rows] = True
    kernel, K_sums = None, None
    for step in range(budget - n_init):
        refit = kernel is None or step % refit_every == 0
        gp = fit_gp(X[rows], np.asarray(yields), seed, kernel=None if refit else kernel)
        if refit:
            kernel, K_sums = gp.kernel_, None
        mu_g, sd_g, K_sums = group_mean_posterior(gp, X, groups, K_sums)
        ei = expected_improvement(mu_g, sd_g, mu_g.max(), xi)
        open_g = np.array([not measured[g].all() for g in groups])
        if not open_g.any():
            break
        g = groups[int(np.argmax(np.where(open_g, ei, -np.inf)))]
        cand = g[~measured[g]]
        _, sd = gp.predict(X[cand], return_std=True)
        pick = int(cand[np.argmax(sd)])
        rows.append(pick)
        yields.append(float(_query_rows(oracle, [pick])[0]))
        measured[pick] = True
    phases = ["init"] * n_init + ["bo"] * (len(rows) - n_init)
    return CampaignResult(oracle.name, "gbo", seed, rows, np.asarray(yields), phases)


def run_screen_bo(oracle: ReactionOracle, X: np.ndarray, budget: int, seed: int,
                  factor: str = "ligand", keep: int = 3) -> CampaignResult:
    """Screen every level of ``factor`` at one random fixed condition, keep the top ``keep`` levels,
    then run BO inside the reduced space with the remaining budget (WP2B-style reduction)."""
    rng = np.random.default_rng(seed)
    anchor = oracle.data.iloc[int(rng.integers(len(oracle)))]
    fixed = {f: anchor[f] for f in oracle.factors if f != factor}
    screen = screen_factor(oracle, factor, fixed)
    kept = top_levels(screen, factor, keep, oracle.target)
    rows, yields = screen.index.tolist(), screen[oracle.target].to_numpy()
    mask = oracle.data[factor].isin(kept).to_numpy()
    n_screen = len(rows)
    rows, yields = _bo_loop(oracle, X, rows, yields, mask, budget - n_screen, seed)
    phases = ["screen"] * n_screen + ["bo"] * (len(rows) - n_screen)
    return CampaignResult(oracle.name, f"screen{keep}_bo", seed, rows, np.asarray(yields), phases,
                          screened_factor=factor, kept_levels=kept)


def summarise(results: list[CampaignResult], oracle: ReactionOracle, top_fraction: float = 0.05) -> pd.DataFrame:
    """Per-run metrics: best yield, global optimum found, and top-5% coverage (EDBO/GOLLuM metric)."""
    y = oracle.data[oracle.target]
    top = set(oracle.top_fraction(top_fraction).index)
    global_max = y.max()
    out = []
    for r in results:
        found = len(top.intersection(r.rows))
        out.append({
            "benchmark": r.benchmark, "strategy": r.strategy, "seed": r.seed,
            "n_experiments": len(r.rows), "best_yield": float(r.yields.max()),
            "found_global_max": bool(np.isclose(r.yields.max(), global_max)),
            "top_found": found, "top_coverage": found / len(top),
            "kept_levels_contain_best": None if r.kept_levels is None
            else bool(oracle.data.loc[y.idxmax(), r.screened_factor] in r.kept_levels),
        })
    return pd.DataFrame(out)


def best_so_far_curves(results: list[CampaignResult]) -> pd.DataFrame:
    """Long table of best-so-far yield after each experiment, for plotting."""
    frames = [pd.DataFrame({"benchmark": r.benchmark, "strategy": r.strategy, "seed": r.seed,
                            "experiment": np.arange(1, len(r.rows) + 1), "best_so_far": r.best_so_far()})
              for r in results]
    return pd.concat(frames, ignore_index=True)
