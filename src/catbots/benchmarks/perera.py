import pandas as pd
from rdkit import Chem

from catbots.benchmarks._common import smiles_column
from catbots.chem import map_unique, remove_fragments
from catbots.oracle import NONE, ReactionOracle
from catbots.paths import BENCHMARK_DIR

INTERNAL_STANDARD = "CCc1cccc(CC)c1"  # 1,3-diethylbenzene, appended to every electrophile SMILES in the source file


def _has_boron(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and any(a.GetSymbol() == "B" for a in mol.GetAtoms())


def load_perera_suzuki(subset: str = "full") -> ReactionOracle:
    """Pfizer Suzuki-Miyaura HTE grid (Perera et al.), as distributed with the GOLLuM benchmarks.

    subset="full": all 5,760 reactions (15 substrate pairs incl. reversed polarity, no-ligand and no-base wells).
    subset="edbo": the 3,696-reaction subspace used by EDBO (boronic-species nucleophiles, ligand and base present).
    """
    if subset not in ("full", "edbo"):
        raise ValueError("subset must be 'full' or 'edbo'")
    raw = pd.read_csv(BENCHMARK_DIR / "perera_suzuki" / "suzuki_miyaura_data.csv")
    electrophile = map_unique(raw["reactant_1_smiles"], lambda s: remove_fragments(s, [INTERNAL_STANDARD]))
    data = pd.DataFrame({
        "electrophile": smiles_column(electrophile),
        "nucleophile": smiles_column(raw["reactant_2_smiles"]),
        "ligand": smiles_column(raw["ligand_smiles"]),
        "base": smiles_column(raw["reagent_1_smiles"]),
        "solvent": smiles_column(raw["solvent_1_smiles"]),
        "yield": 100 * raw["objective"].astype(float),
        "catalyst": raw["catalyst_smiles"],  # Pd(OAc)2 throughout
    })
    if subset == "edbo":
        boron = map_unique(data["nucleophile"], _has_boron)
        data = data[boron & (data["ligand"] != NONE) & (data["base"] != NONE)]
    return ReactionOracle(
        name="perera_suzuki" if subset == "full" else "perera_suzuki_edbo",
        data=data,
        factors=("electrophile", "nucleophile", "ligand", "base", "solvent"),
        citation="Perera et al., Science 2018, 359, 429-434; GOLLuM: Rankovic et al., Nat. Mach. Intell. 2026",
        notes="Single product; the substrate axis varies leaving group / boron species only. Pd(OAc)2 fixed. "
              "The 3 bidentate ligands (dtbpf, dppf, Xantphos) have no Kraken descriptors.",
    )
