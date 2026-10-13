import pandas as pd

from catbots.benchmarks._common import smiles_column
from catbots.oracle import ReactionOracle
from catbots.paths import BENCHMARK_DIR


def load_doyle_arylation() -> ReactionOracle:
    """Imidazole C-H arylation substrate-scope grid: 24 ligands x 8 aryl bromides x 8 imidazoles."""
    raw = pd.read_csv(BENCHMARK_DIR / "doyle_arylation_scope" / "aryl-scope-ligand-augmented-kraken.csv")
    # the CSV also carries pre-merged Kraken columns; featurise with KrakenLigandFeaturizer instead
    data = pd.DataFrame({
        "ligand": smiles_column(raw["SMILES"]),
        "electrophile": smiles_column(raw["electrophile_smiles"]),
        "nucleophile": smiles_column(raw["nucleophile_smiles"]),
        "yield": raw["yield"].astype(float),
        "ligand_name": raw["ligand_name"],
        "electrophile_id": raw["electrophile_id"],
        "nucleophile_id": raw["nucleophile_id"],
        "product": raw["product_smiles"],
    })
    return ReactionOracle(
        name="doyle_arylation",
        data=data,
        factors=("ligand", "electrophile", "nucleophile"),
        citation="Wang et al., Nature 2024, 626, 1025-1033 (bandit optimisation)",
        notes="Other conditions fixed. 7 yields exceed 100% (HPLC calibration); left unclipped.",
    )
