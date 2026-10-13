# Research log: cat_bots computational pipeline

Running record of findings, decisions and results since the project became computational.
Newest experiments are appended at the end of §5. Numbers are reproducible from the scripts named
in each entry (scratch analyses are marked as such).

---

## 1. Project status

- **2026-10-01:** physical SDL experiments were dropped (the target cyclisation chemistry proved too complex).
  The WP1–WP2 statistical pipeline from `project_overview.md` is being turned into an open-source
  computational project, validated retrospectively against published, fully measured HTE datasets
  ("oracles") instead of the robot.
- Research question taking shape: **when does cheap, interpretable statistical reduction of a reaction
  space (screening → models → reduced space) make low-budget BO more efficient, and what molecular
  representations (DFT descriptors, foundation-model embeddings, multimodal) make it work?**

## 2. Benchmark datasets (oracles)

All live in `src/data/reaction_benchmarks/` (see its README for sources, licences and known file issues).
Loaded with `catbots.load_benchmark(name)`; yields in %, SMILES canonicalised, absent components `<none>`.

| Name | Reactions | Factors | Pipeline stages it can test |
|---|---|---|---|
| `shields_arylation` | 1,728, complete grid | ligand 12 × base 4 × solvent 4 × conc 3 × T 3 | ligand screen + BO |
| `doyle_arylation` | 1,536, complete grid | ligand 24 × aryl bromide 8 × imidazole 8 | WP2B (capped) + WP2C + final BO |
| `perera_suzuki` / `_edbo` | 5,760 / 3,696 | substrate pair 15 (12) × ligand 12 (11) × base 8 (7) × solvent 4 | ligand screen + BO |
| `cernak_cn` | 33,792 rows → 29,572 recipes (sparse) | metal complex 29 × ligand 148 × base 11 × solvent 4 × ArX 2 × T 3 | WP2A metal screen + WP2B |

### Dataset findings (scratch analyses, 2026-10-01)

**Doyle C–H arylation** (Wang et al., *Nature* 2024, bandit paper)
- Variance shares: ligand 14%, aryl bromide 23%, imidazole 23%, ArBr × imidazole 14%, ligand × substrate 26%.
- Clear "most general" ligand: Cy-BippyPhos, 46.2% mean over 64 pairs; it is top in 84% of bootstrap
  resamples. Next: Et-PhenCar-Phos 43.9, tBPh-CPhos 43.6, CgMe-PPh 43.0.
- The best ligand depends on the substrate pair: ligand-rank Spearman between pairs has median 0.43; the
  per-pair winner is spread over 10 ligands (CgMe-PPh wins 12 of 64).
- **Kraken descriptors do not predict ligand generality from 22–24 ligands:** LOO R² from −0.4 to −0.7
  (ElasticNet, Ridge, random forest; full 186 or reduced 33 descriptors; 3 targets). Adding Kraken to a
  leave-ligand-out reaction model lowers R² (0.56 → 0.36).
- CgMe-PPh and MeO-KITPHOS *are* in Kraken; the CSV's merge missed them on stereo SMILES (fixed by
  InChIKey matching).

**Perera Suzuki** (Perera et al., *Science* 2018; GOLLuM copy)
- Single product: the "substrate" axis is leaving group (Cl/Br/I/OTf) × boron species only.
- Variance: ligand 12.5%, solvent 3%, base 0.6% (main effects); ligand × substrate pair 13.5%.
- A single-condition ligand screen (keep top 3 = 25% of space) retains top-5% reactions at about the
  random rate, and drops PPh3, which supplies 23% of the top-5%.
- `shields_suzuki.csv` = EDBO 3,696 subset with identical yields; it differs only in SMILES writing
  (BF3K without K+, LiO*i*Pr for LiO*t*Bu, dtbpf/CataCXium A notation).

**Shields direct arylation** (Shields et al., *Nature* 2021, EDBO)
- **Ligand dominates:** 48% of variance (solvent 8%, T 3%, base 1%, conc 0.2%; pairwise interactions 24%).
- A single-condition ligand screen keeping the top 3 retains a median 66/87 top-5% reactions (random ≈ 22)
  and keeps the global best in 82% of screens.
- 11 of its 12 ligands appear in the Doyle set; CgMe-PPh gives the 100% reaction (same chemistry family).

**Cernak C–N** (Das et al., *JACS* 2026)
- Only the 4-X-benzotrifluoride + morpholine pair has yields locally; the 3-bromopyridine pair needs ORD.
- Very sparse hits: 71% zero yield, 1.3% > 20%, 0.1% > 50%; best 99% (Pd + DavePhos).
- Strong metal-free background (NaHMDS controls up to 47%); overall Cu hit rate (> 20%) equals metal-free (0.7%).
- 11 plates, each a complete 8 catalysts (7 + no metal) × 48 ligands × 4 bases × 2 T factorial; metal type
  and solvent are confounded across plates (Cu mostly DMSO, Pd/Ni mostly diglyme).
- Ligands: 49 monophosphines (47 in Kraken), 60 N/O, 19 polyphosphine, 17 NHC.

**Cross-dataset observation (hypothesis for the paper):** single-condition screening works when the
screened factor's main effect dominates (Shields, 48%) and fails when interactions dominate (Suzuki, 12.5%).

## 3. Software and infrastructure

- `catbots` package (`src/catbots/`, `pyproject.toml`), tested with pytest (`tests/`).
  - `oracle.py`: `ReactionOracle` (`query` counts budget; `search_space`, `top_fraction`, `subset`).
  - `benchmarks/`: loaders for the 4 datasets.
  - `featurizers/`:
    - `kraken.py`: InChIKey-block matching; default descriptor set = `kraken_reduced_0.7_no_delta.csv`
      (34 descriptors, same ligand coverage as the full table).
    - `encoding.py`: one-hot / scaled numeric / descriptor blocks.
    - `rdkit_desc.py`: RDKit 2D descriptors for substrates without DFT descriptors.
  - `selection.py`: Kennard-Stone; a regression test shows identical selection order to the WP2 scripts.
  - `preprocessing.py`: the `corr_variance_wp2a.py` greedy correlation filter.
  - `models.py`: WP2B three-model fit (ElasticNet / MLP / GPR, SHAP ranking), WP2C activity classifier,
    multitask ElasticNet.
  - `screening.py`: metal dead-zone mapping against matched no-metal controls; single-factor screens.
  - `bo.py`: GP (ARD Matérn-5/2) + expected improvement, batch size 1.
  - `campaign.py`: simple strategies (random, BO, screen → BO).
  - `pipeline.py`: the overview pipeline (`OverviewBatches`, `Lab` stage-logged experiment ledger).
- **Environment:** `environment.yml` pins `pandas<3`, `pillow<12` and `packaging<26`. Without them,
  streamlit (pip) installs a second, older copy over conda's; the pins mean no package is locked by both
  managers. pytest was added. The lock was re-locked 2026-10-01 (scikit-learn 1.9.1, RDKit 2026.03.1,
  pandas 2.3.3).
- Run scripts with the env's DLLs on PATH (or `conda activate cat-bots`); otherwise numpy's BLAS crashes.
  Set `OMP_NUM_THREADS=1` for parallel experiment runs.
- **BO speed-up (2026-10-01):** `GPExpectedImprovement(refit_every=5)` re-optimises GP hyperparameters
  every 5 suggestions and reuses them (exact GP update) in between. With 40 ARD length scales and
  ~190 points, refitting every step cost ~30 min per campaign (~74 s per Doyle reference pair after the
  change). This applies to every BO arm alike. §5.1 results were produced with refits at every step.
- **GPU env `catbots-gnn` (2026-10-01)** (`environment-gnn.yml`):
  - torch 2.14.1+cu130 on the RTX 5090; the driver supports CUDA 13.2.
  - Pinned to the exact `+cu130` build; otherwise pip picks PyPI's newer CPU-only wheel.
  - chemprop 2.3.1 with RDKit from PyPI (its `cuik_molmaker` extension links the PyPI RDKit DLLs).
  - CheMeleon weights: `~/.chemprop/chemeleon_mp.pt` (Zenodo 15460715; cite arXiv:2506.15792).
    Architecture: BondMessagePassing (d_h = 2048, depth 6), mean aggregation, v2 multi-hot atom
    featuriser, giving 2048-d embeddings.

## 4. Methodological decisions

| Decision | Rationale | Date |
|---|---|---|
| **Budgets follow the overview's stage batches** (KS 60 ligands → first 12 × metal salts → remaining 48 → 12 KS from the predicted top 15% → 36 KS substrates → k-means 5 → 4 representatives × 24 KS ligands → final 8 substrates × 20 ligands → BO), **capped at pool size** | The project is about the overview's small-data regime; literature (EDBO/GOLLuM 10 + 50) and "fraction of space" budgets were rejected | 2026-10-01 |
| **BO budget = 25% of the reduced space** | Overview rule (320–640 = 25% of 1,280–2,560) | 2026-10-01 |
| Stages run only where a dataset has the dimension (Cernak: WP2A–B; Doyle: WP2B–C + BO; Shields/Suzuki: screen + BO) | Datasets lack metals or substrates | 2026-10-01 |
| Baselines (random, plain BO) get exactly the pipeline run's experiment count | Equal-budget comparison | 2026-10-01 |
| "Substrate 1" is swept over every substrate pair instead of being fixed | The overview fixes it by expert choice; a sweep removes that arbitrariness | 2026-10-01 |
| Model choice by LOO-CV R², not the scripts' 80/10/10 split | 12–72 points leave 1–3 test points | 2026-10-01 |
| Metal dead-zone rule compares against matched metal-free controls | Background SNAr/aryne chemistry (Cernak) makes raw yields misleading | 2026-10-01 |
| The original `metal_filtering_wp2a.py` "strict" rule (≥5% for *every* ligand) is not used | It inverts the overview's rule (drop a metal only if < 5% for *all* ligands) | 2026-10-01 |
| Experiment re-runs are free | The oracle has no noise; a lab would not repeat a measured reaction | 2026-10-01 |
| **Factors outside the overview's design (base; Shields also concentration and temperature; Cernak base, temperature and aryl halide) are fixed at the run's reference level in every stage**; solvent is screened at the reference level and opened in the final space | The overview implicitly fixes everything it does not list, and screens at one solvent (anisole) before opening all 4 in the final space | 2026-10-02 |
| Every possible reference condition is run once (Shields 144, Cernak 60, Suzuki 480), as with the 64 Doyle pairs | The overview's reference is an expert choice; the sweep shows how much results depend on it | 2026-10-02 |
| Molecules in the new benchmarks use the "combined" representation (CheMeleon PCA + Kraken/RDKit; Kraken median-imputed with a `kraken_missing` flag for non-Kraken ligands) | Most robust arm in the Doyle ablation (§5.3) | 2026-10-02 |
| Single-substrate benchmarks (Shields, Cernak) use **best yield found** as the objective; benchmarks with substrates use **generality** = true mean yield over all substrates of the recommended condition vector | Generality is undefined without a substrate dimension | 2026-10-02 |
| **WP0 pre-screen (optional, `--prescreen`)**: 30 experiments (3 diverse probes × 10 D-optimal combinations of the undesigned factors + solvent) choose the reference by a ridge main-effects model; counted in the budget, baselines matched | The fixed reference decided the outcome on every benchmark (§5.8); about 30 extra experiments agreed with the user (§5.9) | 2026-10-02 |
| **Final ligands: measured ligands ranked by observed mean over the representatives; the multitask prediction only for unmeasured ligands** | The model's in-sample ranking dropped the best ligand in 52% of Doyle runs (§5.11); deviates from the overview, which ranks all ligands by the model | 2026-10-03 |

## 5. Experiments

### 5.1 `experiments/first_benchmark.py`: screen → BO vs BO (SUPERSEDED; exploratory)

Strategies: random; BO (10 random + EI); BO with Kennard-Stone initial design; screen all ligands at one
random fixed condition → keep top 3 → BO. 100 seeds each. **Budgets did not follow the overview**
(§4), so treat as exploratory.

- **Budget 60** (`results/exploratory_screen_then_bo/budget60/`): screen → BO equals BO on top-5% coverage (Shields
  0.221 vs 0.222, p = 0.86; Suzuki 0.056 vs 0.059, p = 0.32). It is slightly worse on best yield (Suzuki
  97.2 vs 97.8, p = 0.006). The screen keeps the best ligand in 75% (Shields) / 25% (Suzuki) of runs.
- **Budget 0.9% of space** (16 / 34 experiments; `results/exploratory_screen_then_bo/fraction0.0091/`):
  - Shields: screen → BO beats BO on top-5% coverage (0.027 vs 0.017, p = 0.0009); global max 8% vs 2%.
  - Suzuki: it loses (coverage 0.023 vs 0.028, p = 0.015; best 94.9 vs 96.7, p = 0.002).
  - Consistent with the main-effect hypothesis in §2.
- 20 seeds are not enough: plain BO's global-max rate on Shields moved from 80% (20 seeds) to 64% (100 seeds).

### 5.2 `experiments/overview_pipeline_doyle.py`: the overview pipeline on Doyle arylation

Setup:
- Batches capped to 24 ligands / 64 pairs / one metal.
- Ligands described by Kraken (reduced); substrate pairs by RDKit descriptors of both partners
  (|r| < 0.7 filter → 33).
- Run once per reference pair (64 runs). Random and BO baselines get the same experiment count.
- Metric: true mean yield over all 64 pairs of the recommended ligand (generality) and its rank among 24.

Per-run experiment counts (reference pair 0):

| Stage | Experiments |
|---|---|
| WP2A | 12 |
| WP2B-1 | 12 |
| WP2B-2 | 0 (no untested ligands) |
| WP2C-1 | 36 |
| WP2C-2 | 93 (96 minus 3 already measured) |
| BO | 40 |
| **Total** | **193 (12.6% of space)** |

The WP2B models had LOO R² of −0.57 (ElasticNet), −1.48 (MLP) and −0.30 (GPR).

**Results (2026-10-01; `results/doyle/overview_pipeline/`, 64 reference pairs):**
- Mean 187.5 experiments per run (12.2% of the space). By stage: WP2A 12, WP2B-1 12, WP2C-1 36,
  WP2C-2 89.4, BO 40.

| Strategy | Generality of recommended ligand (true mean over 64 pairs) | Median rank of 24 | #1 ligand | Top-3 ligand |
|---|---|---|---|---|
| **Pipeline** | **43.3 ± 3.0** | **2.5** | **30%** | **73%** |
| Random (same n) | 42.6 ± 4.3 (Wilcoxon p = 0.41) | 3 | 25% | 52% (sign test 22 wins / 8 losses, **p = 0.016**) |
| Plain BO (same n) | 41.4 ± 4.7 (**p = 0.007**) | 4 | 22% | 44% (25 / 6, **p = 0.001**) |

- True generality: Cy-BippyPhos 46.2 > Et-PhenCar-Phos 43.9 > tBPh-CPhos 43.6 > CgMe-PPh 43.0 (24-ligand
  mean 30.3). The top is flat (~3 points), so the **top-3 hit rate** is the discriminating metric.
- Pipeline recommendations: Cy-BippyPhos 19, tBPh-CPhos 15, Et-PhenCar-Phos 13, CX-PICy 6, CgMe-PPh 5,
  others 6.
- **Plain BO is worst for generality.** It optimises single-reaction yield, which supports a
  generality-oriented pipeline ahead of (or instead of) a yield-maximising optimiser.
- **The result is robust to the reference ("substrate 1") choice:** Spearman(reference quality, pipeline
  generality) = 0.05. The same picture holds without the 5 dead references (top-3 73% → 71%; vs BO
  p = 0.002) and on the 32 above-median references only (vs BO p = 0.008; vs random n.s., p = 0.42).
- **Clustering collapse:** in 3 runs the classifier's top-5 substrate features separated only 2 groups,
  giving 2 multitask tasks and 2 final substrates (83 experiments instead of ~193). This is the known
  substrate-representation weakness (§6) and a target for CheMeleon / two-tower substrate embeddings.
- Recommendation rules differ by design: the pipeline uses the GP mean over its 8 final substrates;
  baselines use the GP mean over all 64 pairs.
- Re-run on 2026-10-02 with traces under `results/doyle/overview_pipeline/kraken_rdkit/`: identical numbers.
- **Superseded on 2026-10-03 by the final-ligand fix (§5.11).** The numbers above use the multitask model's
  ranking. With the fix, the Kraken + RDKit arm gives 43.5 ± 3.6, #1 ligand 39%, top-3 78% (vs BO p = 0.005,
  top-3 29/7). Old runs: `results/doyle/overview_pipeline/superseded_multitask_model_ranking/`.

### 5.3 Representation ablation on Doyle: Kraken + RDKit vs CheMeleon vs combined (2026-10-02)

Same script, `--features {kraken_rdkit, chemeleon, combined}`, identical overview batches; only the
representation used by the reduction stages changes (KS, WP2B models, classifier, clustering, multitask).
The BO surrogate is one-hot in all arms.
- **CheMeleon:** PCA of 2048-d embeddings to 90% variance, fitted per molecule set → ligands 9 components,
  substrate pairs 6 + 6.
- **Combined:** both representations concatenated → ligands 43, substrates 45.
- Comparison: `experiments/plot_doyle_ablation.py` → `results/doyle/ablation_representation/`.
  Paired by reference pair; baselines are taken from the Kraken + RDKit arm.

Rerun on 2026-10-03 with the final-ligand fix (§5.11); the pre-fix numbers are in the note below the table.

| Arm | Experiments (mean) | Generality (mean ± SD) | #1 ligand | Top-3 ligand | Runs < 40% | Worst | Cluster collapse |
|---|---|---|---|---|---|---|---|
| Pipeline, Kraken + RDKit | 187.5 | 43.5 ± 3.6 | 39% | 78% | 9 | 32.2 | 3 |
| Pipeline, CheMeleon | 193.1 | **44.4 ± 2.3** | **47%** | 77% | 2 | 32.4 | **0** |
| **Pipeline, combined** | 188.8 | 44.1 ± **1.3** | 25% | **86%** | **0** | **41.8** | 2 |
| Random (same n) | 187.5 | 42.6 ± 4.3 | 25% | 52% | 10 | 20.7 | – |
| Plain BO (same n) | 187.5 | 41.4 ± 4.7 | 22% | 44% | 19 | 27.9 | – |

Before the fix, the means were 43.3 / 43.8 / 44.0, #1 ligand 30% / 27% / 19% and top-3 73% / 67% / 86%
(Kraken + RDKit / CheMeleon / combined). The fix helps every arm, most of all CheMeleon (#1 ligand 27% → 47%).

- **Combined vs baselines** (within its own run, budget-matched):
  - vs random: top-3 wins/losses 29/3 (sign test p = 3e-6); generality Wilcoxon p = 1e-4.
  - vs BO: 32/5 (p = 7e-6); generality p = 5e-4.
- **No representation is significantly better than another at 64 references:**
  - combined vs Kraken + RDKit: generality p = 0.91, top-3 13/8 (p = 0.38);
  - CheMeleon vs Kraken + RDKit: generality p = 0.29, top-3 12/13.
- **The representations trade off differently:**
  - combined is the most **robust**: lowest SD, no run below 40%, worst 41.8, best top-3 (86%). It
    concentrates on tBPh-CPhos (true rank 3; 32 of 64 runs), which explains its low #1 rate;
  - CheMeleon gives the best **mean** and finds the single best ligand most often (47%).
- **CheMeleon removes the clustering collapse** (0/64 vs 3/64 for Kraken + RDKit). The embeddings separate
  substrates that the classifier-selected RDKit features could not.
- **Interpretation:** both arms that include CheMeleon beat Kraken + RDKit alone on the mean, on robustness
  (2 runs and 0 runs below 40%, against 9) and on the clustering. This supports the foundation-model /
  multimodal direction (§7), but none of the differences is significant at this sample size.
- Generality-vs-experiment curves (`ablation_generality_vs_experiments.png`) use one common
  recommendation rule (GP over the full space) at every checkpoint, so all arms are comparable at
  intermediate budgets. The final dots use each strategy's own rule.
- CheMeleon cache: `src/data/embeddings/chemeleon_mp.npz`, 252 molecules. It failed to featurise 6, all
  Cernak: 4 solvent *names* (not SMILES) and 2 BF4⁻ salt ligands.

### 5.4 Generalised pipeline (2026-10-02)

`catbots.pipeline.run_overview_pipeline` with a `PipelineSpec` maps a benchmark's factors onto the
overview's roles:
- `ligand`;
- `substrates` (empty means a single substrate, so the objective becomes best yield);
- `solvent`;
- `metal` + `metal_group` (WP2A metal screen against the `<none>` metal-free control; best salt kept per
  active metal);
- `fixed` (held at the reference level).

It also handles sparse grids: the ligand and metal-source universe is restricted to what is measured at
the reference. The Doyle function is now a wrapper around it; on Doyle references 0, 9 and 40 it
reproduces the original implementation's experiment sequence, decisions and recommendation exactly.
Driver: `experiments/overview_pipeline_benchmarks.py --benchmark {shields_arylation, cernak_cn, perera_suzuki}`.

### 5.5 Shields direct arylation: overview pipeline vs baselines (2026-10-02)

Spec: solvent screened then opened; base, concentration and temperature fixed at the reference.
Every one of the 144 reference conditions (4 bases × 3 conc × 3 T × 4 solvents) is run once.
- Per run: WP2A 12 (all 12 ligands at the reference); WP2B-1, WP2B-2 and WP2C are empty (pool exhausted,
  single substrate).
- Final space: 12 ligands (top-20 capped) × 4 solvents = 48 reactions; BO 25% = 12.
  **Total 24 experiments (1.4% of the space).** Baselines also get 24.
- Objective: best yield found. `results/shields/overview_pipeline/combined/`.

| Strategy | Best yield found (mean ± SD) | Found global max (100%) | Best ≥ 50% | Top-1% reactions found | Pipeline wins/losses |
|---|---|---|---|---|---|
| Pipeline | 83.1 ± 15.4 | 5.6% | 94% | 0.46 | – |
| Random | 81.7 ± 13.3 (Wilcoxon p = 0.26) | 4.2% | 97% | 0.31 | 79/62 |
| **Plain BO** | **89.7 ± 9.3 (p = 1e-4)** | 6.3% | 100% | 1.01 | 54/88 |

**Why the pipeline loses on average: the reference, not the pipeline.**
- Inside its reduced space the pipeline finds the best available reaction in **91% of runs**.
  The reduced space's ceiling (best yield that exists in its 48 reactions) is mean 84.2, median 88.1,
  range 38.8–100. The ceiling equals the global max in only 8/144 references.
- Split by that ceiling:

| Ceiling of the reduced space | References | Pipeline | Random | Plain BO | p (pipeline vs BO) |
|---|---|---|---|---|---|
| < 80% | 36 | 62.0 | 78.7 | 88.7 | < 0.001 (BO better) |
| 80–95% | 76 | 86.7 | 82.8 | 90.6 | 0.002 (BO better) |
| **≥ 95%** | **32** | **98.1** | 82.6 | 88.5 | **< 0.001 (pipeline better)** |

- By reference temperature (mean best yield, pipeline vs BO): 120 °C 92.8 vs 89.1; 105 °C 89.0 vs 91.2;
  **90 °C 67.4 vs 88.8**. By reference base, KOAc is the weakest (76.4 vs 91.5) and CsOAc the best
  (90.0 vs 89.5).
- **Conclusion:** at 24 experiments the overview's reduction is very effective *within* the space it
  keeps. Fixing undesigned factors at a reference condition is a gamble that decides the outcome.
  With a chemically sensible reference (high temperature, Cs carboxylate) the pipeline beats plain BO
  by about 10 points; with a poor one (90 °C, KOAc) it is capped well below what BO finds.
- **Implications** (to test): either the reference must be chosen by a principled rule (e.g. a small
  initial screen of the undesigned factors) rather than fixed by assumption, or those factors should be
  reopened in the final space (rejected for now, §4).
- Figures in `results/shields/overview_pipeline/combined/`: `best_so_far_yield.png`,
  `final_best_yield.png`, `pipeline_trace_example.png`, and **`by_reference_ceiling.png`**
  (the split table above, mean ± 95% CI).

### 5.6 Cernak C–N: overview pipeline vs baselines (2026-10-02; rerun with the ferrocene fix)

**Rerun (authoritative)**, `results/cernak/overview_pipeline/combined/`. 145/147 ligands; the first run
is archived in `results/cernak/overview_pipeline/superseded_missing_ferrocenes/`. Per run: WP2A 116.8, WP2B-1 37.0, BO 11.4;
total 159.1 experiments (0.5% of the space). Wall time 78 min on 32 cores, dominated by the plain-BO
baseline (see §7, item 6).

| Strategy | Best yield found (mean ± SD) | Best ≥ 50% | Top-1% reactions found | Pipeline wins/losses |
|---|---|---|---|---|
| Pipeline | 19.5 ± 18.5 | 8% | 2.2 | – |
| Random | 34.7 ± 16.0 (Wilcoxon p = 2e-5) | 15% | 1.7 | 14/46 |
| Plain BO | 38.0 ± 12.7 (p < 1e-6) | 8% | 16.4 | 8/52 |

| What the fixed reference allows (best yield at that base / T / ArX, any ligand or metal) | References | Pipeline | Random | Plain BO | p vs BO |
|---|---|---|---|---|---|
| < 20% | 39 | 8.8 | 34.9 | 36.1 | < 0.001 (BO better) |
| 20–50% | 15 | 32.2 | 33.1 | 42.3 | 0.026 (BO better) |
| **≥ 50%** | **6** | **56.9** | 37.6 | 39.8 | 0.094 (n.s., n = 6) |

- The pipeline reached its reduced-space ceiling in 77% of runs.
- Metals kept by WP2A: Cu 17, Pd 15, Ni 9; two or three metals 19.
- Including the ferrocenes changed nothing material (first run 19.5 / 34.9 / 37.8).
- Figure: `by_reference_ceiling.png`.
- Reading: as on Shields, the pipeline does well inside its reduced space; the reference fixes the outcome.
  Cernak is the harshest case because hits are rare: in 65% of references *no* metal/ligand
  combination exceeds 20%.

The first-run notes below are kept for the bug record.

Spec: metal source = metal complex, with WP2A against the metal-free control and the best complex kept
per active metal. Solvent screened then opened. Base, temperature, aryl halide and amine fixed at the
reference. The 60 reference conditions are those with metal-free controls.
- Per run: WP2A mean 117 (12 ligands × (7–19 complexes + control); 96 for the standard 7-complex
  plates), WP2B-1 32.4, BO 11.5. Total 154 experiments (0.5% of the 29.6k recipes).
- Objective: best yield found.

**First run, SUPERSEDED by a bug:** 11 ligands had no descriptors and were excluded. They are all
ferrocenyl phosphines written with dative bonds (`->[Fe+2]`), including QPhos, one of the paper's best
Pd and Ni ligands. RDKit cannot make an InChIKey for them, so the CheMeleon lookup (keyed by InChIKey)
silently dropped them. Fixed (fallback to canonical SMILES; regression test added); 145/147 ligands now
have descriptors (the 2 BF4⁻ salts genuinely fail Chemprop featurisation). Doyle, Shields and Suzuki
are unaffected (all their ligands have InChIKeys).

First-run numbers, kept only for the reference-dependence pattern:
- Best yield found: pipeline 19.5 ± 18.8; random 34.9 (p = 6e-6, pipeline wins/losses 12/48);
  BO 37.8 (p < 1e-6, 8/52).
- **Same mechanism as Shields:** in 39/60 reference conditions *no* ligand × metal combination exceeds
  20% (reference ceiling median 16.1%), so fixing base/temperature/aryl halide caps the pipeline from
  the start. It reached its reduced-space ceiling in 80% of runs.
  - Reference ceiling < 20% (n = 39): pipeline 8.7 vs BO 35.8.
  - Reference ceiling 20–50% (n = 15): 32.9 vs 42.3.
  - **Reference ceiling ≥ 50% (n = 6): 55.6 vs BO 39.8 and random 38.8.**
- Metals kept by WP2A: Cu only 21, Pd only 15, Ni only 10, two or three metals 14.

### 5.7 Perera Suzuki: overview pipeline vs baselines, generality objective (2026-10-02)

Spec: substrates = (electrophile, nucleophile), 15 pairs; solvent screened then opened; base fixed at
the reference. All 480 references run once (15 pairs × 8 bases incl. "no base" × 4 solvents).
- Ligand pool: 11 real ligands (no-ligand wells excluded from the pool). Combined descriptors; Kraken
  imputed for the 3 bidentates.
- Per run: WP2A 11, WP2C-1 14, WP2C-2 40, BO 80.8. **Total 145.8 experiments (2.5% of 5,760).**
- Final space: 8 pairs × 11 ligands × 4 solvents, base fixed.
- Generality = true mean yield over all 15 pairs of the recommended condition vector
  (ligand, base, solvent); 384 possible vectors; best 67.3%, mean vector 40.1%.
- Results: `results/suzuki/overview_pipeline/combined/`.

| Strategy | Generality (mean ± SD) | Median rank of 384 | #1 vector | Top-5% vector (rank ≤ 19) | vs pipeline |
|---|---|---|---|---|---|
| **Pipeline** | **58.8 ± 6.5** | **21** | 9.6% | **47.9%** | – |
| Random | 57.1 ± 7.3 | 31 | 3.8% | 39.4% | generality p = 7e-4; top-5% 134/93, p = 0.008 |
| Plain BO | 58.5 ± 7.0 | 22 | 11.9% | 46.7% | generality p = 0.37; top-5% 131/125, p = 0.75 |

**Over all references** the pipeline beats random but **ties plain BO**. As on Shields and Cernak, the
outcome is decided by the fixed undesigned factor, here the reference base:

| Reference base | Ceiling (best vector with that base) | Pipeline | Random | Plain BO | p vs BO |
|---|---|---|---|---|---|
| K3PO4 | 67.3 | **64.8** | 57.4 | 58.0 | < 0.001 |
| KOH | 66.7 | **64.2** | 58.0 | 58.6 | < 0.001 |
| NaOH | 65.7 | 60.5 | 57.4 | 58.6 | 0.27 |
| NaHCO3 | 65.2 | 58.7 | 56.8 | 59.6 | 0.59 |
| LiOtBu | 65.2 | **61.4** | 57.5 | 57.5 | 0.001 |
| Et3N | 60.0 | 56.2 | 56.4 | 57.6 | 0.10 |
| CsF | 57.5 | 53.3 | 58.1 | **58.6** | < 0.001 (BO better) |
| no base | 56.4 | 51.6 | 55.3 | **59.0** | < 0.001 (BO better) |

- **References whose base allows a near-best vector** (ceiling within 3 points of 67.3; n = 300):
  pipeline 61.9 vs BO 58.5 (p < 1e-4); **top-5% hit rate 69% vs 46%** (118/48, p < 1e-4).
- **Other bases** (n = 180): pipeline 53.7 vs BO 58.4; top-5% 12% vs 48%.
- The pipeline reaches the best vector available with its base in 35% of runs.
- The generality-vs-experiments curve (common recommendation rule) shows the pipeline ahead of BO and
  random from about experiment 40 (end of WP2C-1) onwards.
- Figures: `generality_vs_experiments.png`, `final_generality.png`, `best_so_far_yield.png`,
  `pipeline_trace_example.png`, **`by_reference_base.png`**.

**Figures written by every `overview_pipeline_doyle.py` run** (`results/doyle/overview_pipeline/<features>/`,
each with a CSV of the plotted values):
- `best_so_far_yield.png`
- `generality_vs_experiments.png`
- `final_generality.png`
- `pipeline_trace_example.png`

### 5.8 Cross-dataset summary: where the overview pipeline helps (2026-10-02)

`experiments/cross_dataset_summary.py` → `results/cross_dataset/summary_table.csv`,
`advantage_vs_reference_ceiling.png`, plus per-dataset `by_reference_ceiling.png` / `by_reference_base.png`.
All runs use the combined representation and overview batches capped at pool size; baselines are
budget-matched.

The **reference ceiling** is the best objective value that exists with the run's fixed factors at the
reference, as % of the global optimum (Doyle has no fixed factors: always 100%).
Δ is pipeline minus baseline in the dataset's metric (yield or generality, %); p from Wilcoxon, paired.

| Dataset (metric) | References | Experiments (coverage) | Δ vs plain BO, all refs | Δ vs random, all refs | Δ vs BO, ceiling ≥ 95% | Δ vs random, ceiling ≥ 95% |
|---|---|---|---|---|---|---|
| Doyle (generality) | 64 | 189 (12.3%) | **+2.2** (40/16, p < 1e-3) | **+2.1** (38/16, p < 1e-3) | same (all refs ≥ 95%) | same |
| Suzuki (generality) | 480 | 146 (2.5%) | +0.4 (256/212, p = 0.37) | **+1.7** (p = 0.001) | **+3.5** (203/87, p < 1e-4; n = 300) | **+4.5** (p < 1e-4) |
| Shields (best yield) | 144 | 24 (1.4%) | **−6.6** (54/88, p < 1e-3) | +1.3 (p = 0.26) | **+9.6** (25/6, p < 1e-3; n = 32) | **+15.5** (27/4, p < 1e-3) |
| Cernak (best yield) | 60 | 159 (0.5%) | **−18.6** (8/52, p < 1e-3) | **−15.2** (p < 1e-3) | n = 1 (only 1 reference ≥ 95%) | – |

**Conclusions:**
1. **When the fixed factors are set well, the pipeline is significantly better than both plain BO and
   random on every dataset that has such references.** Doyle +2.2, Suzuki +3.5, Shields +9.6 points
   vs BO. Within its reduced space it reaches the best available result in 77–91% of single-substrate
   runs.
2. **Averaged over all references, the outcome is decided by the factors the overview leaves undesigned**
   and fixes by assumption: base, concentration, temperature (and aryl halide on Cernak). Its advantage
   over BO rises steeply with the reference ceiling (`advantage_vs_reference_ceiling.png`). It loses on
   Shields (−6.6) and badly on Cernak (−18.6), where in 65% of references nothing exceeds 20% yield.
3. **The pipeline is most useful for generality on substrate-rich problems** (Doyle; Suzuki with a sound
   base) and **for single-substrate optimisation only when the reference condition is good**.
4. **Practical implication for the method:** the overview's assumption that undesigned factors can be
   fixed by expert choice is its main point of failure. Two candidate fixes, to be tested:
   (a) choose the reference by a small initial screen of those factors (e.g. Kennard-Stone over base ×
   temperature at a few ligands) before WP2A; (b) reopen them in the final space (§4, currently rejected).
   **(a) is now implemented and tested as WP0, §5.9.**
5. **Compute cost:** plain BO is far more expensive than the pipeline on large spaces (Cernak: 78 min of
   wall time for 60 references, almost all in the BO baseline); quantified in §5.9.

**Results layout (2026-10-02):**
- `results/<dataset>/<experiment>/`:
  - `doyle/overview_pipeline/{kraken_rdkit,chemeleon,combined}`, `doyle/ablation_representation`;
  - `shields|suzuki|cernak/overview_pipeline/combined` (Cernak also `superseded_missing_ferrocenes`);
  - `shields|suzuki|cernak/overview_pipeline_wp0/combined` (with the WP0 pre-screen, §5.9);
  - `doyle|suzuki/generality_bo_baseline` (§5.10);
  - `cross_dataset/wp0_selection_quality` (§5.9);
  - `doyle/multitask_diagnostic` (§5.11).
- `results/cross_dataset/`.
- `results/exploratory_screen_then_bo/` (the superseded §5.1 runs).

### 5.9 WP0: statistical pre-screen for the reference condition (2026-10-02)

**Why it is needed.** The overview designs ligands, metals, substrates and solvent, but leaves every other
factor (base, temperature, concentration, aryl halide) at an expert-chosen reference. The benchmark sweeps
(§5.5–5.8) showed that this choice decides the outcome:
- On Shields, the pipeline gains +9.6 points over BO when the reference ceiling is ≥ 95%, but loses by −6.6
  averaged over all 144 references.
- On Cernak, 65% of references cannot exceed 20% yield whatever the ligand and metal.
- On Suzuki, a poor base caps generality well below the optimum.

In a real lab, a chemist's prior partly protects against this. On the benchmarks the reference is
arbitrary, so the pipeline needs its own data-driven way to set those factors. Otherwise a comparison
mostly measures the luck of the reference. WP0 is that step: a small designed screen run before WP2A.

**Method** (`src/catbots/prescreen.py`, `select_reference`; stage label `WP0`):
1. **Probes.** 3 chemically diverse probes:
   - each is a Kennard-Stone ligand, with a random KS start per seed;
   - where there is a metal factor, each probe gets a different metal (one representative complex per metal
     group, rotating);
   - where there are substrates, each gets a different KS substrate, starting from substrate 1.

   Spreading probes over ligands, metals and substrates is what lets the choice generalise rather than fit
   one ligand.
2. **Design.** A D-optimal subset of 10 combinations of the screened factors **and the solvent**:
   - main-effects one-hot model with an intercept;
   - greedy forward selection plus Fedorov exchange maximising log det(X'X);
   - restricted to combinations measured with every probe.

   10 × 3 = **30 experiments**. On sparse grids (Cernak), the required number of commonly measured
   combinations is halved until 3 probes fit.
3. **Model.** Ridge regression (α = 1) on condition main effects plus probe offsets.
   - The chosen reference is the candidate with the highest predicted **mean over probes** (probe columns
     set to 1/P), a generality-minded choice.
   - Dead-zone filter: design combinations where every probe gave < 5% are excluded.
   - The screened factors are fixed at the chosen levels in all later stages. The chosen solvent becomes
     WP2's screening solvent; it is still opened in the final space.
4. **Budget.** The 30 WP0 experiments are logged in the same `Lab` the pipeline continues. They count
   towards the total, later stages reuse any WP0 measurement, and the baselines are matched to the larger
   total. This is the one addition to the overview's batches, agreed for this purpose (about 30 experiments).

Screened factors (`SCREEN` in `experiments/overview_pipeline_benchmarks.py`):
- Shields: base × concentration × temperature (+ solvent), 144 candidates.
- Suzuki: base (+ solvent), 32 candidates.
- Cernak: base × temperature (+ solvent), with the aryl halide still given by the reference.

Runs: `--prescreen` → `results/<dataset>/overview_pipeline_wp0/combined/`.
- Shields: 50 seeds.
- Suzuki: 15 reference substrate pairs × 8 seeds = 120 runs.
- Cernak: 2 aryl-halide / amine references × 20 seeds = 40 runs.

**Design lessons from building it:**
- A first version screened at one substrate and the reference solvent. On Suzuki, its base ranking did not
  transfer, so solvent and substrates are now varied inside the design.
- Cernak's sparse grid needed per-ligand metal fallback and relaxed coverage. Otherwise no probe set was
  measured across enough combinations.

**Offline selection quality** (WP0 alone, 10 seeds per distinct reference; `experiments/wp0_selection_quality.py` → `results/cross_dataset/wp0_selection_quality/`).
"Ceiling" is the best value achievable at the chosen fixed levels, as % of the global optimum:

| Dataset | WP0-chosen ceiling (mean) | ≥ 95% | Random reference (mean) | ≥ 95% |
|---|---|---|---|---|
| Shields | **95.0%** | **60%** | 84.2% | 22% |
| Suzuki | 94.8% | 75% | 93.6% | 62% |
| Cernak | 37.9% | 5% | 33.6% | 2% |

**Full pipeline with WP0 vs budget-matched baselines** (combined representation; Wilcoxon, paired):

| Dataset | Runs | Experiments | Pipeline | Random | Plain BO | Pipeline without WP0 (§5.5–5.7) |
|---|---|---|---|---|---|---|
| Shields (best yield) | 50 | 52.5 (30 WP0) | 94.1 ± 5.7 | 89.3 (p = 4e-4; 36/13) | **98.2** (p = 3e-4; 12/34) | 83.1 at 24 experiments (BO 89.7) |
| Suzuki (generality) | 120 | 176.7 (30 WP0) | 57.6 ± 7.1 | 57.6 (p = 0.74) | **59.6** (p = 0.027) | 58.8 at 146 experiments (BO 58.5) |
| Cernak (best yield) | 40 | 210 (30 WP0) | 28.5 ± 16.7 | 36.0 (p = 0.10; 18/22) | **42.1** (p = 2e-4; 12/28) | 19.5 at 159 experiments (BO 38.0) |

Cernak runs: 2 aryl-halide/amine references × 20 seeds. WP0 raised the mean reference ceiling from about 34%
(random reference) to 41.8%, but only 8% of runs reached a ceiling ≥ 95%. The pipeline improved from 19.5 to
28.5 but still trails random and BO. It stays poor even at high ceilings: in the 3 runs with a 100% ceiling
it found only 19.8. So on Cernak the problem is not only the reference. The ligand / metal reduction itself
(WP2A keeps Cu, Ni or Pd subsets after a 127-experiment metal screen) often drops the region where the rare
good yields sit. Most chosen conditions were KOtBu or Cs₂CO₃ in DMSO.

**Conclusions:**
1. **WP0 does what it was built for on additive problems.** On Shields it lifts the chosen reference
   ceiling from 84% to 95% and the pipeline from 83.1 to 94.1, and the pipeline now beats random
   significantly (+4.8).
2. **It does not beat budget-matched BO.** BO also receives the 30 extra experiments. With 52 instead of
   24 experiments, plain BO on Shields' 1,728 reactions reaches 98.2%. At the overview's batch sizes, the
   pre-screen's 30 experiments are worth more to an unconstrained global optimiser than to the
   reduce-then-optimise pipeline.
3. **On Suzuki, WP0 does not help.** The pipeline drops from 58.8 to 57.6 while BO rises from 58.5 to 59.6.
   The diagnosis comes from `runs.csv`, grouping by the chosen base and screening solvent:
   - WP0 picked "no base" in 26 of 120 runs, and Et₃N or CsF in 12 more; those runs reach 50–54%
     generality. The good bases (KOH, K₃PO₄, LiOtBu; 63–64%) were picked in only 38 runs.
   - WP0 picked MeCN as the screening solvent in 72% of runs. Runs screened in MeCN reach 56.8, against
     62.4 in MeOH.
   - At the same base, WP0 runs are slightly worse than no-WP0 runs.

   Suzuki's base effect depends strongly on the boron species (acid vs Bpin vs BF₃K): a base × substrate
   interaction. 3 probes covering 3 of 15 substrate pairs under a main-effects model cannot rank 8 bases
   reliably for the whole scope.
4. **Implication.** A main-effects pre-screen is enough when factors act roughly additively (Shields). It is
   not enough when the undesigned factor interacts with the substrate (Suzuki). On a sparse,
   low-yield landscape (Cernak), even a better reference does not rescue the reduction steps. Candidate improvements, not yet tested:
   - more probes but fewer combinations (e.g. 5 × 6);
   - pick the screening solvent from WP2A data rather than WP0;
   - keep the top-2 WP0 levels and open them in the final space.
5. **Framing for the paper.** With WP0, the pipeline is not a better *single-substrate yield* optimiser
   than budget-matched BO (Shields, Cernak). BO searches all factors jointly and needs no reference. The
   pipeline's case is **generality on substrate-rich problems**:
   - Doyle: +2.2 vs BO with nothing fixed (§5.2);
   - Suzuki: +3.5 vs BO when the base is sound (§5.8); WP0 runs that chose KOH, K₃PO₄ or LiOtBu reach
     63–64%, against BO's 59.6 mean.

   It also gives interpretable intermediate results (SHAP, dead metals), plate-sized batches, and much
   lower compute on large spaces.

   Caveats: the BO baseline maximises yield, not generality, so a generality-aware BO is the stronger
   baseline still to test. The benchmark grids (1.7k–30k) are also far smaller than the overview's ~88k
   space, which favours plain BO.

**Compute cost** (mean wall time per run, single process; next step 6). No speed-ups were applied, at the
user's request:

| Dataset | Pipeline (incl. WP0) | Random | Plain BO | Note |
|---|---|---|---|---|
| Shields | 6.5 s | < 0.01 s | 0.85 s | small space: the pipeline's WP2 model fitting (LOO-CV, SHAP) dominates |
| Suzuki | 201 s | < 0.01 s | 234 s | similar; plain BO runs ~177 GP-EI steps over 5,760 candidates vs the pipeline's ~82 over ~330 |
| Cernak | 311 s | < 0.01 s | 1,851 s | BO over ~29.6k candidates with ~196 one-hot length scales: **6× slower** |

The pipeline costs the same or less than plain BO on medium spaces and is much cheaper on large ones,
because its own BO stage runs over a small final space (Cernak ~37 reactions, Shields 48, Suzuki ~330). On tiny spaces (Shields) the overhead
of the WP2 models makes it slower, but only by seconds.

### 5.10 Generality-aware BO baseline (2026-10-02)

`experiments/generality_bo_baseline.py`; `catbots.campaign.run_generality_bo`; `catbots.bo.group_mean_posterior`.
Output in `results/doyle|suzuki/generality_bo_baseline/`.

**Why.** Plain BO maximises single-reaction yield, but the pipeline is judged on generality. A reviewer
would ask for a BO that targets generality directly.

**Method:**
- One GP over (condition, substrate), as in plain BO: ARD Matern-5/2, refit every 5 steps, 10 random
  initial points.
- Each step picks the **condition** with the highest expected improvement of its posterior *mean over
  all substrates*. The mean and variance of that average use the full within-condition posterior
  covariance; the incumbent is the best predicted mean. The step then measures that condition on its
  **most uncertain unmeasured substrate**.
- Recommendation: the same GP rule as the other baselines (`recommend_general`).
- Run at each cached no-WP0 pipeline run's experiment count, with that run's plain-BO seed. Pipeline,
  random and plain-BO values come from the §5.2 / §5.7 runs.

**Results** (generality %, paired Wilcoxon against the pipeline):

| Dataset | Runs | Pipeline | Random | Plain BO | **Generality BO** | Pipeline wins/losses vs gBO | Top-k hit rate (pipeline / gBO) |
|---|---|---|---|---|---|---|---|
| Doyle (after the §5.11 fix) | 64 | 44.1 | 42.0 | 42.0 | **44.3** (p = 0.41) | 21/24 | top-3 ligand: 86% / 80%; best ligand: 25% / 36% |
| Suzuki, all references | 480 | 58.8 | 57.1 | 58.5 | **63.2** (p < 1e-10) | 143/322 | top-5% condition: 48% / 82% |
| Suzuki, ceiling ≥ 95% | 300 | 61.9 | 57.4 | 58.5 | **62.9** (p = 0.07) | 138/147 | – |

Compute: gBO takes 505 s per Doyle run and 212 s per Suzuki run, about the same as plain BO on Suzuki.

**Conclusions:**
1. **A generality-aware BO matches or beats the overview pipeline on both generality benchmarks.**
   - Doyle: a statistical tie (+0.2, n.s.; +0.3 before the §5.11 fix). The pipeline is slightly more reliable at finding a top-3
     ligand; gBO finds the single best ligand more often.
   - Suzuki: gBO is clearly better overall (+4.4). It needs no reference: base and solvent are searched
     jointly. Even against the pipeline's good-base runs, gBO is level (+1.0, n.s.).
2. **The pipeline's earlier advantage on generality was against a yield-maximising BO.** That advantage
   came from the objective, not from the staged reduction.
3. **What remains for the pipeline:** it is competitive with the best baseline for generality when its
   reference is sound. It is also interpretable (SHAP, metal dead-zones, substrate clusters), works in
   plate-sized batches rather than 1-at-a-time BO, and is cheaper on large spaces. On these benchmarks it is
   **not** a more sample-efficient optimiser than a well-posed BO.

### 5.11 What the WP2C multitask step decides (diagnostic, 2026-10-03)

`experiments/diagnose_multitask_doyle.py` → `results/doyle/multitask_diagnostic/`.

The multitask ElasticNet (one task per substrate-cluster representative) is implemented as in the
overview and runs on every dataset with substrates. Its only decision is which ligands enter the final
space: the top 20 by predicted mean over the representatives. On these benchmarks that decision has almost
no room to act:
- **Suzuki:** 11 ligands < 20, so every ligand is kept. The step's 40 experiments (27% of a run) supply
  data to the final GP, but the model decides nothing.
- **Doyle:** 24 ligands, 20 kept. The step uses **91 of 189 experiments (48%)**, with all 24 ligands × ~5
  representatives. Because every ligand is measured, the model predicts ligands it was trained on, at
  n = 24 with dozens of descriptors.

**It drops the truly most general ligand (Cy-BippyPhos) in 33 of 64 runs (52%).** On average 2.5 of the
true top 3 are kept. The other dropped ligands are mostly ranks 13–23, which is the intended behaviour.
The pipeline still recommends a top-3 ligand in 86% of runs, because ranks 1–3 are within about 3 points,
but this caps how often it can find the single best ligand (19% before the fix, against 36% for generality BO, §5.10).

**Interpretation:**
- In the overview's design (~1,500 ligands, 24 measured per representative), the multitask model
  *extrapolates* to hundreds of unmeasured ligands. That is where it is meant to earn its cost.
- On these benchmarks, the candidate ligand pool is the size of the measured set, so the step is mostly
  an expensive data-collection stage with a noisy in-sample filter.

**Fix (applied 2026-10-03, `pipeline.py` WP2C-2):** ligands measured on every representative are ranked by
their **observed mean** over the representatives; only unmeasured ligands use the multitask prediction.
The budget is unchanged (the final space is still 8 × 20), so the baselines stay valid. Only Doyle is
affected: Suzuki keeps all 11 ligands either way, and Shields and Cernak skip WP2C. All three Doyle arms
were rerun (§5.2–5.3), and the generality-BO comparison (§5.10) and cross-dataset summary (§5.8) were
regenerated.

After the fix, the best ligand is kept in **64/64 runs** (was 31/64) and the true top 3 always survive. The
dropped ligands are now ranks 12–24. The pipeline's #1-ligand rate rises in every arm (combined 19% → 25%,
Kraken + RDKit 30% → 39%, CheMeleon 27% → 47%).

The combined arm's remaining gap to the best ligand comes from the recommendation step: the GP mean over
the 8 final substrates prefers tBPh-CPhos (rank 3, 2.6 points below the best). Pre-fix diagnostic:
`results/doyle/multitask_diagnostic/superseded_multitask_model_ranking/`.

## 6. Known limitations and open issues

- Multitask ElasticNet: substrates are tasks, so it cannot select substrate descriptors or predict unseen
  substrates. Final-space reclustering therefore uses the WP2C classifier's top-5 substrate features;
  those can fail to separate substrates (k is capped at the number of distinct feature vectors).
- WP2B models are not predictive on Doyle at n = 24 (negative LOO R²).
- Cernak pair 2 yields need the ORD download; licence decisions are still open before release
  (Cernak article CC-BY-NC-ND).
- `src/ligand_screening/model_training_wp2c.py` uses `penalty="elasticnet"`, which is deprecated in
  scikit-learn 1.8 and removed in 1.10; the package port (`models.py`) already uses the new API.
- Old scripts in `src/` have inconsistent working-directory assumptions and the path bugs listed at
  project review. They are kept for reference; the package versions supersede them.

## 7. Next steps

1. ~~CheMeleon frozen embeddings as a featuriser; Doyle ablation~~ (done, §5.3: combined is most robust, CheMeleon alone has the best mean; n.s.).
   Follow-ups: more references or bootstrap for power on combined vs Kraken; CheMeleon embeddings in
   the BO surrogate too (currently one-hot).
2. Multimodal product-kernel GP: k_ligand × k_substrate × k_metal × k_conditions.
3. Two-tower ligand × substrate model replacing the multitask ElasticNet; its substrate embeddings
   drive the reclustering.
4. GP-likelihood fine-tuning of embeddings (GOLLuM-style with CheMeleon), after 1–3.
5. ~~Cernak WP2A–B path (metal screen with controls), and Shields/Suzuki through the overview's batches.~~
   (done, §5.5–5.7)
6. ~~Compute-cost comparison~~ (done, §5.9: pipeline 311 s vs BO 1,851 s per Cernak run; similar on
   Suzuki; the pipeline is slower only on tiny Shields, 6.5 s vs 0.85 s).
7. ~~Principled reference selection~~ (WP0 built and tested, §5.9). Follow-ups:
   - WP0 variants: more probes and fewer combinations; screening solvent chosen from WP2A; top-2 WP0
     levels opened in the final space;
   - ~~a generality-aware BO baseline~~ (done, §5.10: gBO ties the pipeline on Doyle, beats it on Suzuki);
   - diagnose why Cernak's metal / ligand reduction misses good regions even at a 100% ceiling.