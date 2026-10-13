"""The project-overview pipeline (WP2A-C + final BO) run against a ReactionOracle.

Batch sizes are the overview's absolute numbers, capped at the size of the pool that exists in a
benchmark (e.g. "Kennard-Stone 60 ligands" screens all 24 ligands of the Doyle arylation set).
Stages whose dimension a benchmark lacks are skipped and recorded as such.

``run_overview_pipeline`` is the general form (``PipelineSpec`` maps a benchmark's factors onto the
overview's roles). Factors outside the overview's design are held at the run's reference levels
throughout; the solvent is fixed while screening and opened in the final space, as in the overview.
``run_ligand_substrate_pipeline`` is the original Doyle-only implementation, kept unchanged so the
recorded Doyle results stay reproducible; ``tests/test_overview_pipeline.py`` checks the two agree.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from catbots.bo import GPExpectedImprovement, fit_gp
from catbots.models import fit_activity_classifier, fit_multitask, select_regressor
from catbots.oracle import ReactionOracle
from catbots.preprocessing import standardise
from catbots.selection import kennard_stone


@dataclass(frozen=True)
class OverviewBatches:
    ligand_pool: int = 60              # WP2A: Kennard-Stone ligands from the stock/commercial set
    first_ligands: int = 12            # WP2A: first KS ligands tested against every metal salt
    refine_top_fraction: float = 0.15  # WP2B-2: top predicted fraction of untested ligands ...
    refine_ligands: int = 12           # ... from which KS picks this many
    substrate_screen: int = 36         # WP2C: KS substrates tested at the best conditions
    substrate_clusters: int = 5        # WP2C: k-means on active substrates (reference + 4 new)
    multitask_ligands: int = 24        # WP2C: KS ligands from the pool run on each new cluster rep
    final_substrates: int = 8          # final space: k-means representatives
    final_ligands: int = 20            # final space: top ligands by multitask average prediction
    bo_fraction: float = 0.25          # final BO budget as a fraction of the reduced space
    active_threshold: float = 5.0      # % yield separating active from inactive


class Lab:
    """Oracle wrapper that logs every experiment with its stage and never repeats a measured reaction."""

    def __init__(self, oracle: ReactionOracle):
        self.oracle = oracle
        self.measured: dict[int, float] = {}
        self.log: list[tuple[int, str, float]] = []
        self._row = {tuple(r): i for i, r in enumerate(oracle.data[list(oracle.factors)].itertuples(index=False))}

    def row(self, **levels) -> int:
        return self._row[tuple(levels[f] for f in self.oracle.factors)]

    def has(self, **levels) -> bool:
        """Whether this combination was measured in the benchmark (sparse grids such as Cernak)."""
        return tuple(levels[f] for f in self.oracle.factors) in self._row

    def run(self, rows: list[int], stage: str) -> np.ndarray:
        new = [r for r in dict.fromkeys(rows) if r not in self.measured]
        if new:
            ys = self.oracle.query(self.oracle.data.loc[new, list(self.oracle.factors)])
            for r, y in zip(new, ys):
                self.measured[r] = float(y)
                self.log.append((r, stage, float(y)))
        return np.array([self.measured[r] for r in rows])

    def get(self, rows: list[int]) -> np.ndarray:
        """Yields of reactions already run; raises instead of silently spending budget."""
        missing = [r for r in rows if r not in self.measured]
        if missing:
            raise KeyError(f"{len(missing)} reactions requested from the log were never run")
        return np.array([self.measured[r] for r in rows])

    @property
    def n_experiments(self) -> int:
        return len(self.log)

    def stage_counts(self) -> dict[str, int]:
        return pd.Series([s for _, s, _ in self.log]).value_counts(sort=False).to_dict()


@dataclass
class PipelineResult:
    reference: str
    lab: Lab
    decisions: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    recommended: str | None = None

    @property
    def n_experiments(self) -> int:
        return self.lab.n_experiments


def _cluster_representatives(F: pd.DataFrame, k: int, seed: int, forced: str | None = None) -> list[str]:
    """One substrate per k-means cluster (closest to centroid); ``forced`` represents its own cluster."""
    if len(F) <= k:
        return F.index.tolist()
    Z = standardise(F).to_numpy()
    # selected features may not separate every substrate (e.g. they only describe one coupling partner),
    # so cap k at the number of distinct feature vectors
    k = min(k, len(np.unique(Z.round(10), axis=0)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    dist = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    labels = pd.Series(km.labels_, index=F.index)
    reps = []
    for c in range(k):
        members = labels.index[labels == c]
        if members.empty:
            continue
        if forced is not None and forced in members:
            reps.append(forced)
        else:
            reps.append(members[np.argmin(dist[labels.to_numpy() == c])])
    return reps


def run_ligand_substrate_pipeline(
    oracle: ReactionOracle,
    ligand_desc: pd.DataFrame,
    substrate_desc: pd.DataFrame,
    substrate_rows: dict[str, dict],
    reference: str,
    X_space: np.ndarray,
    batches: OverviewBatches = OverviewBatches(),
    ligand_factor: str = "ligand",
    seed: int = 0,
) -> PipelineResult:
    """Single-metal ligand x substrate pipeline (Doyle arylation): ``run_overview_pipeline`` with every
    non-ligand factor a substrate factor. ``recommended`` is the recommended ligand level."""
    spec = PipelineSpec(ligand=ligand_factor, substrates=tuple(f for f in oracle.factors if f != ligand_factor))
    res = run_overview_pipeline(oracle, spec, {}, ligand_desc, X_space, substrate_desc=substrate_desc,
                                substrate_rows=substrate_rows, reference_substrate=reference,
                                batches=batches, seed=seed)
    res.recommended = res.recommended[ligand_factor]
    return res


def recommend_general(oracle: ReactionOracle, X_space: np.ndarray, rows: list[int], yields: np.ndarray,
                      factor="ligand", seed: int = 0):
    """Baseline recommendation: the condition with the highest GP-predicted mean over the whole space.

    ``factor`` is one factor name (returns its level) or a list of condition factors (returns a tuple),
    the mean being taken over everything else (the substrates).
    """
    gp = fit_gp(X_space[rows], yields, seed)
    mu = pd.Series(gp.predict(X_space), index=oracle.data.index)
    keys = factor if isinstance(factor, str) else list(factor)
    return mu.groupby([oracle.data[k] for k in ([keys] if isinstance(keys, str) else keys)]).mean().idxmax()


@dataclass(frozen=True)
class PipelineSpec:
    """How a benchmark's factors map onto the overview's roles.

    ``substrates``: factors that together identify a substrate (empty: single substrate, so the objective
    becomes best yield instead of generality). ``solvent``: screened at the reference level, opened in the
    final space. ``metal``: metal-source factor screened in WP2A; its ``NONE`` level is the metal-free
    control and ``metal_group`` is a data column naming the metal of each source.
    ``fixed``: factors outside the overview's design, held at the reference level in every stage.
    ``screen``: the subset of ``fixed`` whose reference levels the WP0 pre-screen chooses
    (``catbots.prescreen.select_reference``); the rest of ``fixed`` (e.g. the aryl halide) is given.
    """
    ligand: str = "ligand"
    substrates: tuple[str, ...] = ()
    solvent: str | None = None
    metal: str | None = None
    metal_group: str | None = None
    fixed: tuple[str, ...] = ()
    screen: tuple[str, ...] = ()

    @property
    def objective(self) -> str:
        return "generality" if self.substrates else "yield"

    def condition_factors(self) -> list[str]:
        """Factors forming a recommended condition vector (everything except the substrate)."""
        return [f for f in (self.ligand, self.metal, self.solvent, *self.fixed) if f is not None]

    def validate(self, oracle: ReactionOracle) -> None:
        roles = [r for r in (self.ligand, *self.substrates, self.solvent, self.metal, *self.fixed) if r is not None]
        if sorted(roles) != sorted(oracle.factors):
            raise ValueError(f"spec roles {roles} must cover the oracle factors {list(oracle.factors)} exactly")
        if not set(self.screen) <= set(self.fixed):
            raise ValueError("screen factors must be a subset of the fixed factors")


def run_overview_pipeline(
    oracle: ReactionOracle,
    spec: PipelineSpec,
    reference: dict,
    ligand_desc: pd.DataFrame,
    X_space: np.ndarray,
    substrate_desc: pd.DataFrame | None = None,
    substrate_rows: dict[str, dict] | None = None,
    reference_substrate: str | None = None,
    batches: OverviewBatches = OverviewBatches(),
    seed: int = 0,
    lab: Lab | None = None,
) -> PipelineResult:
    """WP2A (metal screen vs metal-free controls) -> WP2B -> WP2C (if substrates) -> final space -> BO.

    ``reference``: levels of the solvent and fixed factors used while screening.
    ``reference_substrate``: the overview's substrate 1 (a key of ``substrate_rows``).
    ``lab``: an experiment ledger that already holds earlier experiments (e.g. the WP0 pre-screen); they
    count towards the budget and are reused instead of re-run.
    """
    from catbots.oracle import NONE

    spec.validate(oracle)
    b = batches
    lab = lab if lab is not None else Lab(oracle)
    res = PipelineResult(reference_substrate or "single substrate", lab)
    d, diag = res.decisions, res.diagnostics
    if spec.substrates:
        if substrate_desc is None or substrate_rows is None or reference_substrate is None:
            raise ValueError("substrate benchmarks need substrate_desc, substrate_rows and reference_substrate")
    else:
        substrate_rows, reference_substrate = {"_": {}}, "_"
    ref = reference_substrate
    fixed_ref = {f: reference[f] for f in spec.fixed}

    def levels(lig, sub, metal=None, solvent=None):
        lv = {spec.ligand: lig, **substrate_rows[sub], **fixed_ref}
        if spec.metal:
            lv[spec.metal] = metal
        if spec.solvent:
            lv[spec.solvent] = reference[spec.solvent] if solvent is None else solvent
        return lv

    def rows_for(ligs, subs, metals=(None,), solvents=(None,)):
        return [lab.row(**levels(l, s, m, v)) for s in subs for m in metals for v in solvents for l in ligs]

    def yields_at(sub, ligs, metal=None):
        return pd.Series(lab.get(rows_for(ligs, [sub], [metal])), index=list(ligs))

    # ---- ligand / metal-source universe measured at the reference condition ----
    lig_all = [l for l in ligand_desc.index if not ligand_desc.loc[l].isna().any()]
    if spec.metal:
        sources = [m for m in oracle.levels(spec.metal) if any(lab.has(**levels(l, ref, m)) for l in lig_all)]
        while True:  # drop the least-covered source until enough ligands are measured with every source
            universe = [l for l in lig_all if all(lab.has(**levels(l, ref, m)) for m in sources)]
            if len(universe) >= min(b.first_ligands, len(lig_all)) or len(sources) <= 2:
                break
            cover = {m: sum(lab.has(**levels(l, ref, m)) for l in lig_all) for m in sources if m != NONE}
            sources.remove(min(cover, key=cover.get))
        metal_sources, has_control = [m for m in sources if m != NONE], NONE in sources
    else:
        universe = [l for l in lig_all if lab.has(**levels(l, ref))]
        metal_sources, has_control = [None], False
    if len(universe) < 2:
        raise ValueError("fewer than 2 ligands measured at the reference condition")
    d["ligand_universe"], d["metal_sources_screened"] = universe, metal_sources
    Zl = standardise(ligand_desc.loc[universe])

    # ---- WP2A: KS ligand pool; first ligands x every metal source (+ metal-free control) ----
    pool = [universe[i] for i in kennard_stone(Zl.to_numpy(), min(b.ligand_pool, len(universe)))]
    first = pool[: b.first_ligands]
    d["ligand_pool"], d["first_ligands"] = pool, first
    lab.run(rows_for(first, [ref], metal_sources + ([NONE] if has_control else [])), "WP2A")
    if spec.metal:
        table = pd.DataFrame({m: yields_at(ref, first, m) for m in metal_sources})
        effect = table.sub(yields_at(ref, first, NONE), axis=0) if has_control else table
        group = (oracle.data.groupby(spec.metal)[spec.metal_group].first().loc[metal_sources]
                 if spec.metal_group else pd.Series(metal_sources, index=metal_sources))
        kept = []
        for _, srcs in pd.Series(group.index, index=group.to_numpy()).groupby(level=0):
            srcs = list(srcs)
            if effect[srcs].max().max() >= b.active_threshold:  # dead-zone rule, against matched controls
                kept.append(table[srcs].mean().idxmax())       # best salt per active metal
        if not kept:
            kept = [table.mean().idxmax()]
            diag["wp2a_note"] = "no metal beat its control by the threshold; kept the best source"
        diag["wp2a_max_effect_by_source"] = effect.max().round(2).to_dict()
        d["metals_kept"] = sorted({group[m] for m in kept})
    else:
        kept = [None]
    d["metal_sources_kept"] = kept

    # ---- WP2B-1: remaining pool ligands with the kept metal sources; low-data models ----
    lab.run(rows_for(pool[b.first_ligands:], [ref], kept), "WP2B-1")
    ref_table = pd.DataFrame({m: yields_at(ref, pool, m) for m in kept})
    y_pool = ref_table[ref_table.mean().idxmax()]
    choice = None
    if y_pool.std() > 0:
        choice = select_regressor(ligand_desc.loc[pool], y_pool.to_numpy(), seed)
        diag["wp2b1_loo_r2"], diag["wp2b1_model"] = choice.loo_r2, choice.name
        diag["wp2b1_top_features"] = choice.feature_ranking.index[:5].tolist()
    else:
        diag["wp2b1_model"] = "skipped: identical yields for every pool ligand"

    # ---- WP2B-2: KS among the top predicted untested ligands ----
    untested = [l for l in universe if l not in pool]
    picks = []
    if untested and choice is not None:
        pred = pd.Series(choice.model.predict(ligand_desc.loc[untested]), index=untested)
        top = pred.nlargest(max(1, math.ceil(b.refine_top_fraction * len(untested)))).index.tolist()
        picks = [top[i] for i in kennard_stone(Zl.loc[top].to_numpy(), min(b.refine_ligands, len(top)))]
        lab.run(rows_for(picks, [ref], kept), "WP2B-2")
    else:
        diag["wp2b2"] = "skipped: every measurable ligand is already in the pool"
    d["refined_ligands"] = picks
    ref_table = pd.DataFrame({m: yields_at(ref, pool + picks, m) for m in kept})
    d["best_ligand"], d["best_metal_source"] = ref_table.stack().idxmax()
    best_lig, best_src = d["best_ligand"], d["best_metal_source"]

    if spec.substrates:
        # ---- WP2C-1: KS substrates at the best conditions; activity classifier ----
        subs = [s for s in substrate_desc.index if lab.has(**levels(best_lig, s, best_src))]
        others = [s for s in subs if s != ref]
        Zs = standardise(substrate_desc.loc[subs])
        screened = [others[i] for i in kennard_stone(Zs.loc[others].to_numpy(), min(b.substrate_screen, len(others)))]
        lab.run(rows_for([best_lig], screened, [best_src]), "WP2C-1")
        train = [ref] + screened
        y_sub = pd.Series(lab.get(rows_for([best_lig], train, [best_src])), index=train)
        clf = fit_activity_classifier(substrate_desc.loc[train], (y_sub >= b.active_threshold).to_numpy(), seed=seed)
        rest = [s for s in subs if s not in train]
        pred_active = clf.predict(substrate_desc.loc[rest]) if rest else np.array([], int)
        active = [s for s in train if y_sub[s] >= b.active_threshold] + \
                 [s for s, a in zip(rest, pred_active) if a == 1]
        d["substrates_screened"], d["active_substrates"] = screened, active
        diag["classifier_features"] = clf.features
        sub_feats = clf.features or substrate_desc.columns.tolist()

        # ---- WP2C-2: cluster representatives x KS ligands from the pool; multitask model ----
        cluster_pool = list(dict.fromkeys([ref] + active))
        reps = _cluster_representatives(substrate_desc.loc[cluster_pool, sub_feats], b.substrate_clusters, seed,
                                        forced=ref)
        new_reps = [s for s in reps if s != ref]
        mt_ligs = [pool[i] for i in kennard_stone(Zl.loc[pool].to_numpy(), min(b.multitask_ligands, len(pool)))]
        lab.run(rows_for(mt_ligs, new_reps, [best_src]), "WP2C-2")
        tasks = [ref] + new_reps
        Y_mt = pd.DataFrame({s: yields_at(s, mt_ligs, best_src) for s in tasks})
        mt = fit_multitask(ligand_desc.loc[mt_ligs], Y_mt, seed)
        # rank by the model's mean over the representatives, but ligands measured on every representative
        # keep their observed mean: the model only has to extrapolate to the unmeasured ones (research log §5.11)
        avg = mt.predict(ligand_desc.loc[universe]).mean(axis=1)
        avg.loc[mt_ligs] = Y_mt.mean(axis=1)
        diag["final_ligands_rule"] = "observed mean over representatives where measured, else multitask prediction"
        d["cluster_representatives"], d["multitask_ligands"] = tasks, mt_ligs
        d["final_ligands"] = avg.nlargest(min(b.final_ligands, len(avg))).index.tolist()
        diag["multitask_top_features"] = mt.feature_ranking.index[:5].tolist()
        d["final_substrates"] = _cluster_representatives(substrate_desc.loc[cluster_pool, sub_feats],
                                                         b.final_substrates, seed)
    else:
        diag["wp2c"] = "skipped: single substrate"
        best_per_ligand = ref_table.max(axis=1)
        d["final_ligands"] = best_per_ligand.nlargest(min(b.final_ligands, len(best_per_ligand))).index.tolist()
        d["final_substrates"] = [ref]

    # ---- Final space: substrates x ligands x kept metal sources x all solvents; fixed factors at reference ----
    solvents = oracle.levels(spec.solvent) if spec.solvent else [None]
    final = [(s, l, m, v) for s in d["final_substrates"] for l in d["final_ligands"] for m in kept for v in solvents
             if lab.has(**levels(l, s, m, v))]
    final_rows = [lab.row(**levels(l, s, m, v)) for s, l, m, v in final]
    d["final_space_size"], d["final_rows"] = len(final_rows), final_rows

    # ---- Final BO: 25% of the reduced space ----
    bo = GPExpectedImprovement(random_state=seed)
    cand = np.array(final_rows)
    for _ in range(round(b.bo_fraction * len(final_rows))):
        avail = cand[[r not in lab.measured for r in cand]]
        if avail.size == 0:
            break
        obs = np.array(list(lab.measured))
        pick = int(avail[bo.suggest(X_space[obs], np.array([lab.measured[r] for r in obs]), X_space[avail])])
        lab.run([pick], "BO")

    # ---- Recommendation: general condition vector, or best reaction for single-substrate benchmarks ----
    if spec.objective == "generality":
        obs = np.array(list(lab.measured))
        gp = fit_gp(X_space[obs], np.array([lab.measured[r] for r in obs]), seed)
        grid = oracle.data.loc[final_rows, spec.condition_factors()].copy()
        grid["mu"] = gp.predict(X_space[final_rows])
        best = grid.groupby(spec.condition_factors())["mu"].mean().idxmax()
        res.recommended = dict(zip(spec.condition_factors(), best if isinstance(best, tuple) else (best,)))
    else:
        best_row = max(lab.measured, key=lab.measured.get)
        res.recommended = oracle.data.loc[best_row, list(oracle.factors)].to_dict()
    return res
