"""The "combined" representation that was most robust in the Doyle ablation (research log §5.3):
CheMeleon PCA components + DFT/2D descriptors, for ligands and substrates alike."""

from collections.abc import Sequence

import pandas as pd

from catbots.featurizers.chemeleon import CheMeleonFeaturizer
from catbots.featurizers.kraken import KrakenLigandFeaturizer
from catbots.featurizers.rdkit_desc import rdkit_descriptors
from catbots.preprocessing import drop_correlated


def combined_ligand_descriptors(ligands: Sequence[str], variance: float = 0.9) -> pd.DataFrame:
    """CheMeleon PCA (fitted on this ligand set) + Kraken (reduced) descriptors.

    Kraken only covers monodentate phosphines; for other ligands its columns are median-imputed and
    a ``kraken_missing`` flag is added. Ligands without a CheMeleon embedding are dropped (with a note
    in ``df.attrs["dropped"]``).
    """
    cm = CheMeleonFeaturizer()
    emb = cm.transform(ligands)
    ok = emb.index[emb.notna().all(axis=1)].tolist()
    pcs = cm.transform_pca(ok, variance=variance, prefix="lig_cm_pc")
    kr = KrakenLigandFeaturizer("reduced").transform(ok)
    missing = kr.isna().any(axis=1)
    out = pd.concat([pcs, kr.fillna(kr.median())], axis=1)
    if missing.any():
        out["kraken_missing"] = missing.astype(float).to_numpy()
    out = out.loc[:, out.notna().all()]  # Kraken columns that are missing for every ligand
    out.attrs["dropped"] = [s for s in ligands if s not in ok]
    return out


def combined_molecule_descriptors(smiles: Sequence[str], prefix: str, variance: float = 0.9,
                                  corr_threshold: float = 0.7) -> pd.DataFrame:
    """RDKit 2D descriptors (correlation-filtered) + CheMeleon PCA for a set of substrate molecules."""
    smiles = list(dict.fromkeys(smiles))
    rd = rdkit_descriptors(smiles)
    rd = rd[drop_correlated(rd, threshold=corr_threshold)] if len(smiles) > 2 else rd
    pcs = CheMeleonFeaturizer().transform_pca(smiles, variance=variance, prefix="cm_pc")
    return pd.concat([rd, pcs], axis=1).add_prefix(f"{prefix}_")
