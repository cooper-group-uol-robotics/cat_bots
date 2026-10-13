"""Screening steps of the reduction pipeline (WP2A metal dead-zone mapping, WP2B single-factor screens)."""

from collections.abc import Sequence

import pandas as pd

from catbots.oracle import NONE, ReactionOracle


def metal_activity(
    data: pd.DataFrame,
    match_on: Sequence[str],
    metal_col: str = "metal",
    target: str = "yield",
    threshold: float = 5.0,
) -> pd.DataFrame:
    """Dead-zone mapping of metals against matched metal-free controls.

    For every condition tuple in ``match_on`` (everything except the metal source) the treatment effect is
    the metal-catalysed yield minus the mean yield of the matching ``metal == NONE`` controls. A metal is
    active if any of its reactions beats its control by at least ``threshold`` yield points; this is the
    overview's "<5% across all tests -> eliminate" rule, corrected for background (SNAr/aryne) reactivity.
    """
    controls = data[data[metal_col] == NONE].groupby(list(match_on))[target].mean().rename("control_yield")
    treated = data[data[metal_col] != NONE].join(controls, on=list(match_on))
    treated = treated.dropna(subset=["control_yield"])
    treated["effect"] = treated[target] - treated["control_yield"]
    out = treated.groupby(metal_col).agg(
        n_matched=("effect", "size"),
        max_yield=(target, "max"),
        max_effect=("effect", "max"),
        frac_effect_above=("effect", lambda e: (e >= threshold).mean()),
    )
    out["active"] = out["max_effect"] >= threshold
    return out.sort_values("max_effect", ascending=False)


def screen_factor(oracle: ReactionOracle, factor: str, fixed: dict) -> pd.DataFrame:
    """Run one experiment per level of ``factor`` with every other factor held at ``fixed``.

    Experiments go through ``oracle.query`` so they count against the budget.
    """
    others = [f for f in oracle.factors if f != factor]
    mask = pd.Series(True, index=oracle.data.index)
    for f in others:
        mask &= oracle.data[f] == fixed[f]
    rows = oracle.data.loc[mask, list(oracle.factors)]
    if rows.empty:
        raise KeyError(f"no measured reactions at the fixed condition {fixed}")
    out = rows.copy()
    out[oracle.target] = oracle.query(rows)
    return out


def top_levels(screen: pd.DataFrame, factor: str, k: int, target: str = "yield") -> list:
    """The ``k`` best levels of ``factor`` in a screen (ties broken by order of appearance)."""
    ranked = screen.groupby(factor, sort=False)[target].max().sort_values(ascending=False, kind="stable")
    return ranked.index[:k].tolist()
