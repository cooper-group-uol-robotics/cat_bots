import pandas as pd

from catbots.benchmarks._common import fix_mojibake, smiles_column
from catbots.oracle import NONE, ReactionOracle
from catbots.paths import BENCHMARK_DIR

FACTORS = ("metal_complex", "ligand", "base", "solvent", "electrophile", "nucleophile", "temperature")


def load_cernak_cn(aggregate_replicates: bool = True) -> ReactionOracle:
    """Pd/Ni/Cu C-N coupling ultraHTE set (Das et al.), substrate pair 4-X-benzotrifluoride + morpholine.

    Only this pair has yields in the local CSV; the 3-bromopyridine + 4-phenylpiperidine pair (16,896 rows)
    has none and must be obtained from the Open Reaction Database
    (ord_dataset-805ad863feef48579d95d86a728035f4).
    Components are keyed by SMILES: about 1.3% of rows carry a base *name* that contradicts the base SMILES.
    """
    raw = pd.read_csv(BENCHMARK_DIR / "cernak_cn" / "cn_coupling_data.csv", low_memory=False)
    raw = raw[raw["metal_types"].notna() & raw["Assay Yield Sum"].notna()]
    data = pd.DataFrame({
        "metal_complex": smiles_column(raw["Cat_SMILES"]),
        "ligand": smiles_column(raw["Lig_SMILES"]),
        "base": smiles_column(raw["Base_SMILES"]),
        "solvent": raw["solvent_name"].str.strip(),  # mixtures (80:20) are only distinguishable by name
        "electrophile": smiles_column(raw["Elec_SMILES"]),
        "nucleophile": smiles_column(raw["Nuc_SMILES"]),
        "temperature": raw["temperature"].astype(float),
        "yield": 100 * raw["Assay Yield Sum"].astype(float),
        "metal": raw["metal_types"].replace({"_Blank_metal": NONE}),
        "metal_name": raw["metal_name"].replace({"_Blank_metal": NONE}),
        "ligand_name": raw["ligand_name"].map(fix_mojibake).replace({"_Blank_ligand": NONE}),
    })
    if aggregate_replicates:
        meta = data.groupby(list(FACTORS))[["metal", "metal_name", "ligand_name"]].first()
        stats = data.groupby(list(FACTORS))["yield"].agg(["mean", "std", "size"])
        stats.columns = ["yield", "yield_sd", "n_replicates"]
        data = stats.join(meta).reset_index()
    return ReactionOracle(
        name="cernak_cn",
        data=data,
        factors=FACTORS,
        citation="Das et al., J. Am. Chem. Soc. 2026, 148, 26401-26412",
        notes="Sparse hits (71% zero yield, 1.3% > 20%). Includes matched metal-free controls (metal == NONE); "
              "strong bases give metal-free background product. Metal type and solvent are confounded across "
              "plates (Cu mostly in DMSO, Pd/Ni mostly in diglyme).",
    )
