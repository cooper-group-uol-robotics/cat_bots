# cat_bots: statistical reaction-space reduction, benchmarked on public HTE data

[![tests](https://github.com/cooper-group-uol-robotics/cat_bots/actions/workflows/tests.yml/badge.svg)](https://github.com/cooper-group-uol-robotics/cat_bots/actions/workflows/tests.yml)

`catbots` is a Python package implementing the staged reaction-space reduction pipeline of
`project_overview.md`:

> Kennard-Stone ligand selection → metal screen → ligand models → substrate screen and clustering →
> multitask model → reduced final space → Bayesian optimisation.

The pipeline was planned for a self-driving lab. Here it is **validated in simulation**: published,
fully measured high-throughput experimentation (HTE) datasets act as the "robot". Every run is compared
with random sampling, plain BO and generality-aware BO, each given the same number of experiments.

- **Results and conclusions:** [`docs/results_summary.md`](docs/results_summary.md) (summary) and
  [`docs/research_log.md`](docs/research_log.md) (every number, decision and date).
- **Datasets, known file issues and licences:**
  [`src/data/reaction_benchmarks/README.md`](src/data/reaction_benchmarks/README.md).

---

## 1. Repository layout

| Path | Contents |
|---|---|
| `src/catbots/` | The package: oracles, featurisers (Kraken, RDKit, CheMeleon, combined), Kennard-Stone, WP2 models, GP-BO, the overview pipeline, the WP0 pre-screen, plotting |
| `src/data/reaction_benchmarks/` | Benchmark datasets used as oracles (Doyle, Shields, Perera Suzuki, Cernak) |
| `src/data/ligands/` | Kraken descriptor tables (default: `kraken_reduced_0.7_no_delta.csv`) |
| `src/data/embeddings/` | Cached CheMeleon embeddings (`chemeleon_mp.npz`), so the core env needs no PyTorch |
| `experiments/` | Benchmark scripts; each writes to `results/<dataset>/<experiment>/` |
| `results/` | All figures (`.png`, each with a `.csv` of the plotted data), run tables and per-run JSON caches |
| `tests/` | `pytest` suite |
| `docs/` | Results summary and research log |
| `src/ligand_screening/`, `src/substrate_screening/`, `src/utils/` | The original WP2 scripts, kept for reference (superseded by the package) |

---

## 2. Setup

Tested on Windows 11 with Anaconda. The core environment is fully locked in `conda-lock.yml` (win-64).

### 2.1 Core environment (`cat-bots`): everything except recomputing embeddings

```
git clone https://github.com/cooper-group-uol-robotics/cat_bots
cd cat_bots

python -m pip install --upgrade conda-lock
conda-lock install --name cat-bots conda-lock.yml
conda activate cat-bots

python -m pip install -e .[dev]     # the catbots package, editable
python -m pytest                    # 40 tests, ~2 min
```

Troubleshooting:
- **Anaconda Terms-of-Service error during `conda-lock install`:** conda is adding the `defaults` channel
  implicitly. Point it at conda-forge only:
  ```
  echo channels: [conda-forge] > %TEMP%\condarc.yml
  set CONDARC=%TEMP%\condarc.yml
  conda-lock install --name cat-bots conda-lock.yml
  ```
- **numpy crashes on import (Windows, exit code `0xc06d007f`)** when calling the env's `python.exe`
  without activating the env: activate it, or put `<env>\Library\bin` on `PATH`.
- For parallel benchmark runs, set `OMP_NUM_THREADS=1` (the scripts parallelise over runs with joblib).

**Without conda** (any OS, Python ≥ 3.10): `python -m pip install -e .[dev]` in a fresh virtual environment
installs everything the package and tests need from PyPI. CI tests exactly this on Linux and Windows. The
conda environment additionally provides Jupyter, `conda-lock` and the dashboard's `streamlit`.

### 2.2 GPU environment (`catbots-gnn`): only to recompute CheMeleon embeddings

The embeddings are cached in `src/data/embeddings/chemeleon_mp.npz`, so this environment is **not needed**
to run the pipeline or reproduce the results. It is only needed after adding new molecules.

```
conda env create -f environment-gnn.yml
conda activate catbots-gnn
python -m pip install -e . --no-deps
python experiments/compute_chemeleon_embeddings.py   # downloads the CheMeleon weights (Zenodo 15460715) on first use
```

Notes:
- The torch pin (`2.14.1+cu130`) targets RTX 50-series GPUs and needs a driver supporting CUDA 13. On other
  hardware, change it to a matching wheel.
- RDKit must come from PyPI in this environment, not conda-forge: Chemprop's `cuik_molmaker` needs it.

### 2.3 Adding packages (team workflow)

1. Install locally (`conda install -n cat-bots <pkg>`, or `pip install <pkg>` if it is pip-only).
2. Add it to `environment.yml`, under `dependencies`, or under `pip:` for pip-only packages.
3. Re-lock: `conda-lock lock -f environment.yml -p win-64`, then commit `environment.yml` and `conda-lock.yml`.
4. Everyone else runs `conda-lock install --name cat-bots conda-lock.yml`.

Keep the `pandas<3`, `pillow<12` and `packaging<26` pins in `environment.yml`. They match streamlit's pip
requirements; without them pip installs duplicate, older copies.

---

## 3. Running the pipeline

### 3.1 Quick start: one campaign from the command line

```
python experiments/run_pipeline.py --benchmark shields_arylation
```

This runs the pipeline once on a bundled benchmark, then random search and plain BO with the same number of
experiments, and prints the experiments per stage, the recommended conditions and each strategy's result.
Nothing needs editing: the factor roles and reference conditions of every benchmark are predefined.

| Option | Effect |
|---|---|
| `--benchmark {shields_arylation,doyle_arylation,perera_suzuki,cernak_cn}` | Which dataset acts as the lab |
| `--reference N` | Which reference condition (substrate pair, or fixed base/temperature/…) to run; default 0 |
| `--list-references` | Print the available reference conditions and exit |
| `--prescreen` | Choose the fixed factors with the 30-experiment WP0 pre-screen first (not Doyle) |
| `--no-baselines` | Skip random search and BO |

Time per campaign, including baselines: Shields ~5 s, Doyle and Suzuki ~1–2 min, Cernak ~35 min (almost all of it
plain BO over 29.6k candidates). Nothing is written to `results/`.

### 3.2 From Python (e.g. on your own dataset)

```python
from catbots import NONE, load_benchmark
from catbots.featurizers import combined_ligand_descriptors, encode_conditions
from catbots.pipeline import PipelineSpec, run_overview_pipeline
from catbots.prescreen import select_reference

oracle = load_benchmark("shields_arylation")   # also: doyle_arylation, perera_suzuki, cernak_cn
ligand_desc = combined_ligand_descriptors([l for l in oracle.levels("ligand") if l != NONE])
X = encode_conditions(oracle.search_space(), oracle.factors)   # one-hot encoding for the BO stages

# Roles of the factors: the solvent is screened at the reference, then opened in the final space;
# the "fixed" factors stay at the reference level throughout.
spec = PipelineSpec(solvent="solvent", fixed=("base", "concentration", "temperature"))
reference = oracle.data.iloc[0][["base", "concentration", "temperature", "solvent"]].to_dict()

result = run_overview_pipeline(oracle, spec, reference, ligand_desc, X)
print(result.n_experiments, result.lab.stage_counts(), result.recommended)
# 24 {'WP2A': 12, 'BO': 12} {...recommended conditions...}

# Optional WP0 pre-screen: choose the fixed factors (and screening solvent) from 30 designed experiments
spec_wp0 = PipelineSpec(solvent="solvent", fixed=spec.fixed, screen=spec.fixed)
pre, lab = select_reference(oracle, spec_wp0, reference, ligand_desc)
result = run_overview_pipeline(oracle, spec_wp0, pre.reference, ligand_desc, X, lab=lab)
```

- `PipelineSpec` maps a dataset's columns onto the overview's roles: `ligand`, `substrates`, `solvent`,
  `metal` / `metal_group`, `fixed` and `screen`. With substrates, the objective is **generality**: the
  mean yield over all substrates of one condition vector. Without them it is the best yield.
- Stage batch sizes follow the overview (`OverviewBatches`: KS 60 ligands, 12 × metals, 48, 12, 36
  substrates, 5 clusters, 4 × 24, final 8 substrates × 20 ligands, BO = 25% of the final space). They are
  capped at the pool size.
- Every oracle query is counted (`oracle.n_queries`), and `result.lab.log` lists every experiment with its
  stage.
- See `experiments/overview_pipeline_benchmarks.py` (`setup`) for the spec of each benchmark, including
  Cernak's metal screen with metal-free controls.

Baselines on the same oracle: `catbots.campaign.run_random`, `run_bo` and `run_generality_bo`.

### 3.3 Reproducing the results

Run from the repo root in the `cat-bots` environment. Finished runs are cached as JSON under each
`per_reference/` folder, so re-running only redraws figures and tables; delete a folder to recompute.
Wall times are for 32 cores.

| Command | What it does | Output | Time |
|---|---|---|---|
| `python experiments/overview_pipeline_doyle.py --features {kraken_rdkit,chemeleon,combined}` | Pipeline vs random vs BO on Doyle (64 reference substrates), per representation | `results/doyle/overview_pipeline/<features>/` | ~25 min each |
| `python experiments/plot_doyle_ablation.py` | Representation ablation figures | `results/doyle/ablation_representation/` | seconds |
| `python experiments/overview_pipeline_benchmarks.py --benchmark {shields_arylation,perera_suzuki,cernak_cn}` | Pipeline vs baselines over every reference condition | `results/{shields,suzuki,cernak}/overview_pipeline/combined/` | < 1 min / ~70 min / ~80 min |
| `... overview_pipeline_benchmarks.py --benchmark <name> --prescreen` | The same with the WP0 pre-screen | `results/<dataset>/overview_pipeline_wp0/combined/` | < 1 min / ~45 min / ~2 h |
| `python experiments/wp0_selection_quality.py` | How good are WP0's chosen references | `results/cross_dataset/wp0_selection_quality/` | seconds |
| `python experiments/generality_bo_baseline.py --dataset {doyle,suzuki}` | Generality-aware BO at the pipeline runs' budgets (needs the pipeline runs above) | `results/{doyle,suzuki}/generality_bo_baseline/` | ~40 / ~55 min |
| `python experiments/diagnose_multitask_doyle.py` | What the WP2C multitask step keeps and drops | `results/doyle/multitask_diagnostic/` | ~3 min |
| `python experiments/cross_dataset_summary.py` | Cross-dataset table and figures | `results/cross_dataset/` | seconds |

`experiments/first_benchmark.py` is an early exploratory run (superseded; see research log §5.1).

---

## 4. PCA visualisation dashboard (original tooling)

```
cd src/utils
streamlit run dashboard.py
```
