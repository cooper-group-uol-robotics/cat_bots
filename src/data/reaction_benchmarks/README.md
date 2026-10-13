# Reaction benchmarks

Measured HTE reaction grids used as experiment oracles in place of the self-driving lab.
Load them with `catbots.load_benchmark(name)`; loaders canonicalise SMILES, encode absent
components as `<none>` and report yields in percent (0–100).

| Name | Folder | Reactions | Factors | Source |
|---|---|---|---|---|
| `shields_arylation` | `shields_arylation/` | 1,728 (complete grid) | ligand 12 × base 4 × solvent 4 × concentration 3 × temperature 3 | Shields et al., *Nature* 2021, 590, 89 |
| `doyle_arylation` | `doyle_arylation_scope/` | 1,536 (complete grid) | ligand 24 × aryl bromide 8 × imidazole 8 | Wang et al., *Nature* 2024, 626, 1025 |
| `perera_suzuki` | `perera_suzuki/` | 5,760 | 15 substrate pairs × ligand 12 × base 8 × solvent 4 | Perera et al., *Science* 2018, 359, 429 (GOLLuM copy) |
| `perera_suzuki_edbo` | `perera_suzuki/` | 3,696 (complete grid) | EDBO subspace of the above | Shields et al. 2021 |
| `cernak_cn` | `cernak_cn/` | 33,792 rows → 29,572 unique recipes | metal complex 29 × ligand 148 × base 11 × solvent 4 × 2 aryl halides × temperature 3 (sparse) | Das et al., *JACS* 2026, 148, 26401 |

## Known issues in the source files

- **perera_suzuki**: every electrophile SMILES contains the internal standard 1,3-diethylbenzene; the
  loader strips it. `shields_suzuki.csv` is the same experiments as the EDBO subset (yields identical) but
  writes some components differently (BF3K without K+, LiO*i*Pr for LiO*t*Bu, other dtbpf/CataCXium A
  SMILES), so it is kept for reference only.
- **cernak_cn**: only the 4-X-benzotrifluoride + morpholine pair has yields. The 3-bromopyridine +
  4-phenylpiperidine pair (16,896 rows) has none; get it from the Open Reaction Database
  (`ord_dataset-805ad863feef48579d95d86a728035f4`). About 1.3% of rows have a base name that contradicts
  the base SMILES (the loader keys on SMILES). One row is malformed and dropped. Regioisomer yields
  exist for only 1,478 rows.
- **doyle_arylation**: 7 yields exceed 100% (HPLC calibration); the bundled Kraken columns missed
  CgMe-PPh and MeO-KITPHOS because of stereo SMILES. `KrakenLigandFeaturizer` matches on the InChIKey
  connectivity block and finds both.

## Licences

The datasets are redistributed here for benchmarking, with attribution, under their sources' licences
(checked 2026-10-12). They are **not** covered by this repository's MIT licence:

| Dataset | Obtained from | Licence |
|---|---|---|
| `shields_arylation`, `perera_suzuki/shields_suzuki.csv` | EDBO, github.com/b-shields/edbo | MIT |
| `doyle_arylation_scope` (Kraken columns added) | github.com/doyle-lab-ucla/bandit-optimization; release on Zenodo, doi:10.5281/zenodo.8181283 | GPL-3.0 (GitHub); CC-BY 4.0 (Zenodo release) |
| `perera_suzuki/suzuki_miyaura_data.csv` | GOLLuM, github.com/schwallergroup/gollum (Ranković et al.) | CC-BY 4.0 (article) |
| `cernak_cn` | Open Reaction Database, `ord_dataset-805ad863feef48579d95d86a728035f4` (also github.com/cernak-lab/cernakHTE) | CC-BY-SA 4.0 (ORD data); the article text is CC-BY-NC-ND 4.0 |

Please cite the original papers (table above) when using these data.
The `paper_*.txt` files are git-ignored.
