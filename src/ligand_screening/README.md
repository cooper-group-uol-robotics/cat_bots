# Original WP2 ligand-screening scripts (legacy)

These are the standalone scripts written for the planned self-driving-lab work. They are kept for reference
and are superseded by the `catbots` package (`src/catbots/`). They use paths relative to this folder.

## How the WP2A ligand set was produced

The ligand tables in `src/data/ligands/` came from these steps:

1. Remove the Kraken columns that are not meaningful as descriptors (the "delta" columns) → `kraken_no_delta.csv`.
2. Run `corr_variance_wp2a.py` (correlation and variance screening of the descriptors).
3. Run `ligand_screen_stock_comm_wp2a.py` → `stock_commercial_kraken_match.csv`. Its inventory inputs are not
   distributed, so this step cannot be rerun from the repository.
4. Remove unstable ligands from `stock_commercial_kraken_match.csv` by hand → `final_ligand_dataset.csv`.
5. Run `kennard_stone_wp2a.py` → `ks60.csv` (the 60-ligand Kennard-Stone testing pool).
