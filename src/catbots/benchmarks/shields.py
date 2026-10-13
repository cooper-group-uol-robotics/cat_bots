import pandas as pd

from catbots.benchmarks._common import smiles_column
from catbots.oracle import ReactionOracle
from catbots.paths import BENCHMARK_DIR


def load_shields_arylation() -> ReactionOracle:
    """Pd-catalysed direct arylation of an imidazole (EDBO reaction 3), complete 1,728-reaction grid."""
    raw = pd.read_csv(BENCHMARK_DIR / "shields_arylation" / "shields_arylation.csv")
    data = pd.DataFrame({
        "ligand": smiles_column(raw["Ligand_SMILES"]),
        "base": smiles_column(raw["Base_SMILES"]),
        "solvent": smiles_column(raw["Solvent_SMILES"]),
        "concentration": raw["Concentration"].astype(float),
        "temperature": raw["Temp_C"].astype(float),
        "yield": raw["yield"].astype(float),
    })
    return ReactionOracle(
        name="shields_arylation",
        data=data,
        factors=("ligand", "base", "solvent", "concentration", "temperature"),
        citation="Shields et al., Nature 2021, 590, 89-96",
        notes="Fixed substrates and Pd source. Single measurements (no replicates).",
    )
