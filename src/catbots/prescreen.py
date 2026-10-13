"""WP0: statistical choice of the reference condition for factors the overview does not design.

The overview fixes base, temperature, concentration (and similar) by expert choice. On the benchmark
datasets that assumption decided the outcome (research log §5.5-5.8), so WP0 chooses those levels from data:

1. **Probes:** ``n_probes`` chemically diverse probes, each a Kennard-Stone ligand; with a metal factor,
   paired with a different metal (one representative complex per metal, rotating); with substrates,
   paired with a different Kennard-Stone substrate starting from substrate 1.
2. **Design:** a D-optimal subset of ``n_design`` combinations of the screened factors **and the solvent**
   (main-effects one-hot model), restricted to combinations measured with every probe:
   ``n_design x n_probes`` experiments (~30). Varying solvent and probes inside the design matters:
   on Perera Suzuki the base ranking at a single substrate and solvent does not transfer
   (research log §5.9).
3. **Model:** ridge regression on condition main effects + probe offsets. The chosen reference is the
   combination with the highest predicted *mean over probes*, a generality-minded choice; the screened
   factors are fixed at it and the solvent becomes the screening solvent of WP2. Dead-zone filter:
   design combinations where every probe gave < 5% are excluded.

The experiments are logged as stage "WP0" in a ``Lab`` that the main pipeline then continues, so they count
towards the budget and are reused by later stages.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from catbots.oracle import NONE, ReactionOracle
from catbots.pipeline import Lab, PipelineSpec
from catbots.preprocessing import standardise
from catbots.selection import kennard_stone


@dataclass(frozen=True)
class PrescreenBatches:
    n_probes: int = 3
    n_design: int = 10          # 10 combinations x 3 probes = 30 experiments
    ridge_alpha: float = 1.0
    dead_zone: float = 5.0      # % yield; design combinations below this for every probe are excluded


@dataclass
class PrescreenResult:
    reference: dict             # full reference: given levels + the chosen screened levels
    chosen: dict                # the chosen screened levels only
    probes: list[tuple]         # (ligand, metal source or None, substrate key)
    design: pd.DataFrame        # screened-factor combinations that were run
    observed: pd.DataFrame      # design x probe yields
    predicted: pd.Series        # predicted mean over probes for every candidate combination
    n_candidates: int


def _onehot(combos: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(combos.astype(str), dtype=float)


def d_optimal(X: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    """Greedy forward selection + Fedorov exchange maximising log det(X'X) (X includes an intercept)."""
    N, p = X.shape
    n = min(n, N)
    ridge = 1e-6 * np.eye(p)

    def logdet(idx):
        return np.linalg.slogdet(X[idx].T @ X[idx] + ridge)[1]

    chosen = [int(rng.integers(N))]
    while len(chosen) < n:
        cands = [int(i) for i in rng.permutation(N) if i not in chosen]
        chosen.append(max(cands, key=lambda i: logdet(chosen + [i])))
    for _ in range(5):  # exchange passes until no swap improves the design
        improved, cur = False, logdet(chosen)
        for k in range(n):
            for i in rng.permutation(N):
                if int(i) in chosen:
                    continue
                trial = chosen[:k] + [int(i)] + chosen[k + 1:]
                s = logdet(trial)
                if s > cur + 1e-9:
                    chosen, cur, improved = trial, s, True
        if not improved:
            break
    return chosen


def select_reference(
    oracle: ReactionOracle,
    spec: PipelineSpec,
    base_reference: dict,
    ligand_desc: pd.DataFrame,
    substrate_rows: dict[str, dict] | None = None,
    reference_substrate: str | None = None,
    substrate_desc: pd.DataFrame | None = None,
    batches: PrescreenBatches = PrescreenBatches(),
    seed: int = 0,
    lab: Lab | None = None,
) -> tuple[PrescreenResult, Lab]:
    """Run WP0 and return the chosen reference plus the ledger holding its experiments.

    ``base_reference`` gives the levels of the non-screened fixed factors (e.g. the aryl halide); its
    values for the screened factors and the solvent are ignored, since WP0 chooses them.
    """
    if not spec.screen:
        raise ValueError("spec.screen is empty: nothing for WP0 to choose")
    b, rng = batches, np.random.default_rng(seed)
    lab = lab if lab is not None else Lab(oracle)
    design_factors = list(spec.screen) + ([spec.solvent] if spec.solvent else [])
    given = {f: base_reference[f] for f in spec.fixed if f not in spec.screen}
    if not spec.substrates:
        substrate_rows, reference_substrate = {"_": {}}, "_"
    # probe substrates: substrate 1 first, then Kennard-Stone-diverse others
    if spec.substrates and substrate_desc is not None:
        keys = substrate_desc.index.tolist()
        sub_order = [keys[i] for i in kennard_stone(standardise(substrate_desc).to_numpy(), len(keys),
                                                    initial=[keys.index(reference_substrate)])]
    else:
        sub_order = [reference_substrate]

    def levels(lig, metal, sub, combo):
        lv = {spec.ligand: lig, **substrate_rows[sub], **given, **combo}
        if spec.metal:
            lv[spec.metal] = metal
        return lv

    # candidate combinations of the design factors with the given factors at their reference levels
    mask = pd.Series(True, index=oracle.data.index)
    for f, v in given.items():
        mask &= oracle.data[f] == v
    combos = oracle.data.loc[mask, design_factors].drop_duplicates().reset_index(drop=True)
    combo_dicts = combos.to_dict("records")

    # probes: KS-diverse ligands (random KS start per seed), rotating metals and substrates
    ligs = [l for l in ligand_desc.index if not ligand_desc.loc[l].isna().any()]
    order = [ligs[i] for i in kennard_stone(standardise(ligand_desc.loc[ligs]).to_numpy(), len(ligs),
                                            initial=[int(rng.integers(len(ligs)))])]
    if spec.metal:
        rows = oracle.data[mask & (oracle.data[spec.metal] != NONE)]
        per_metal = rows.groupby(spec.metal_group)[spec.metal].agg(lambda s: s.value_counts().index[0])
        metals = list(rng.permutation(per_metal.to_numpy()))
    else:
        metals = [None]
    def pick_probes(need):
        probes, ok = [], np.ones(len(combos), bool)
        for lig in order:
            if len(probes) == b.n_probes:
                break
            sub = sub_order[len(probes) % len(sub_order)]
            k = len(probes) % len(metals)  # preferred metal rotates; others tried if it is not measured
            for metal in metals[k:] + metals[:k]:
                measured = np.array([lab.has(**levels(lig, metal, sub, c)) for c in combo_dicts])
                if (ok & measured).sum() >= need:
                    probes.append((lig, metal, sub))
                    ok &= measured
                    break
        return probes, ok

    # sparse grids (Cernak): relax the required number of commonly measured combinations if needed
    need = min(b.n_design, len(combos))
    probes, ok = pick_probes(need)
    while len(probes) < b.n_probes and need > 2:
        need = max(2, need // 2)
        probes, ok = pick_probes(need)
    if not probes:
        raise ValueError("no probe ligand is measured across enough screened-factor combinations")
    cand = combos[ok].reset_index(drop=True)
    cand_dicts = cand.to_dict("records")

    # D-optimal design over the main-effects one-hot model (with intercept)
    X = np.hstack([np.ones((len(cand), 1)), _onehot(cand).to_numpy()])
    design_idx = d_optimal(X, b.n_design, rng)
    design = cand.iloc[design_idx].reset_index(drop=True)
    for c in design.to_dict("records"):
        lab.run([lab.row(**levels(lig, metal, sub, c)) for lig, metal, sub in probes], "WP0")

    # ridge main-effects model with probe offsets; predict the mean over probes for every candidate
    obs = pd.DataFrame([[lab.measured[lab.row(**levels(lig, metal, sub, c))] for lig, metal, sub in probes]
                        for c in design.to_dict("records")], columns=[f"probe{k + 1}" for k in range(len(probes))])
    Xc = _onehot(cand)
    rows_X, y = [], []
    for i, di in enumerate(design_idx):
        for k in range(len(probes)):
            rows_X.append(np.concatenate([Xc.iloc[di].to_numpy(), np.eye(len(probes))[k]]))
            y.append(obs.iloc[i, k])
    model = Ridge(alpha=b.ridge_alpha).fit(np.array(rows_X), np.array(y))
    pred = model.predict(np.hstack([Xc.to_numpy(), np.full((len(cand), len(probes)), 1 / len(probes))]))
    predicted = pd.Series(pred, index=pd.MultiIndex.from_frame(cand.astype(str)))
    dead = {tuple(map(str, design.iloc[i].tolist())) for i in range(len(design)) if (obs.iloc[i] < b.dead_zone).all()}
    allowed = [i for i in range(len(cand)) if tuple(map(str, cand.iloc[i].tolist())) not in dead]
    best = max(allowed or range(len(cand)), key=lambda i: pred[i])
    chosen = cand_dicts[best]
    reference = {**base_reference, **given, **chosen}  # screened factors + screening solvent chosen
    return PrescreenResult(reference, chosen, probes, design, obs, predicted, len(cand)), lab
