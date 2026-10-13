"""Lookup oracle over a fully or partially measured reaction table.

A ``ReactionOracle`` stands in for the self-driving lab: every "experiment" an
optimisation strategy requests is answered by looking up the measured yield.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NONE = "<none>"  # placeholder for an absent component (no ligand, no base, no metal)


@dataclass
class ReactionOracle:
    name: str
    data: pd.DataFrame
    factors: tuple[str, ...]
    target: str = "yield"  # percent, 0-100 (a few HTE values sit slightly above 100)
    citation: str = ""
    notes: str = ""
    n_queries: int = field(default=0, init=False)

    def __post_init__(self):
        self.factors = tuple(self.factors)
        missing = [c for c in (*self.factors, self.target) if c not in self.data.columns]
        if missing:
            raise ValueError(f"{self.name}: columns missing from data: {missing}")
        if self.data[list(self.factors)].isna().any().any():
            raise ValueError(f"{self.name}: factor columns contain NaN; encode absent components as {NONE!r}")
        dup = self.data.duplicated(list(self.factors))
        if dup.any():
            raise ValueError(f"{self.name}: {dup.sum()} duplicated factor combinations; aggregate replicates first")
        self.data = self.data.reset_index(drop=True)
        self._lookup = self.data.set_index(list(self.factors))[self.target]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        levels = ", ".join(f"{f}={self.data[f].nunique()}" for f in self.factors)
        grid = "complete grid" if self.is_complete_grid else f"{self.coverage:.1%} of full grid"
        return f"ReactionOracle({self.name!r}, n={len(self)}, {levels}; {grid})"

    def levels(self, factor: str) -> list:
        return sorted(self.data[factor].unique().tolist(), key=str)

    @property
    def grid_size(self) -> int:
        return int(np.prod([self.data[f].nunique() for f in self.factors], dtype=np.int64))

    @property
    def coverage(self) -> float:
        return len(self) / self.grid_size

    @property
    def is_complete_grid(self) -> bool:
        return len(self) == self.grid_size

    def search_space(self) -> pd.DataFrame:
        """All measured factor combinations (the candidates an optimiser may query)."""
        return self.data[list(self.factors)].copy()

    def query(self, conditions: pd.DataFrame | dict) -> np.ndarray:
        """Return measured yields for the requested conditions; raises KeyError for unmeasured ones."""
        if isinstance(conditions, dict):
            conditions = pd.DataFrame([conditions])
        keys = pd.MultiIndex.from_frame(conditions[list(self.factors)])
        found = keys.isin(self._lookup.index)
        if not found.all():
            raise KeyError(f"{(~found).sum()} requested conditions were never measured, e.g. {keys[~found][0]}")
        self.n_queries += len(keys)
        return self._lookup.loc[keys].to_numpy()

    def reset_budget(self) -> None:
        self.n_queries = 0

    def top_fraction(self, fraction: float = 0.05) -> pd.DataFrame:
        """Reactions at or above the (1 - fraction) yield quantile, e.g. the top-5% set used by EDBO/GOLLuM."""
        threshold = self.data[self.target].quantile(1 - fraction)
        return self.data[self.data[self.target] >= threshold]

    def subset(self, name: str | None = None, **fixed) -> "ReactionOracle":
        """Restrict factors to given levels, e.g. ``oracle.subset(base=["CsOAc SMILES"])``."""
        mask = pd.Series(True, index=self.data.index)
        for col, allowed in fixed.items():
            allowed = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            mask &= self.data[col].isin(allowed)
        return ReactionOracle(
            name=name or f"{self.name}[subset]",
            data=self.data[mask].copy(),
            factors=self.factors,
            target=self.target,
            citation=self.citation,
            notes=self.notes,
        )
