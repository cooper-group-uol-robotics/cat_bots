"""CheMeleon foundation-model embeddings, read from the cache written by
experiments/compute_chemeleon_embeddings.py (so the core package needs no PyTorch).

CheMeleon: Chemprop BondMessagePassing pretrained to predict Mordred descriptors (arXiv:2506.15792).
"""

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from catbots.chem import canonical_smiles, inchikey14, map_unique
from catbots.paths import EMBEDDING_DIR

DEFAULT_PATH = EMBEDDING_DIR / "chemeleon_mp.npz"


def _key(smiles) -> str | None:
    """InChIKey connectivity block, or the canonical SMILES where RDKit cannot make an InChI
    (e.g. ferrocenyl phosphines written with dative bonds)."""
    return inchikey14(smiles) or canonical_smiles(smiles)


@lru_cache(maxsize=None)
def _load(path: str) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    df = pd.DataFrame(z["X"].astype(float), index=[_key(s) for s in z["smiles"]])
    df.columns = [f"chemeleon_{i}" for i in range(df.shape[1])]
    return df[df.index.notna()].groupby(level=0).mean()


class CheMeleonFeaturizer:
    """Map SMILES to 2048-d CheMeleon embeddings (matched by InChIKey connectivity block)."""

    def __init__(self, path: str | Path = DEFAULT_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} missing: run experiments/compute_chemeleon_embeddings.py "
                                    "in the catbots-gnn environment")
        self.table = _load(str(path))

    def transform(self, smiles: Iterable[str]) -> pd.DataFrame:
        smiles = pd.Series(list(smiles), dtype=object)
        out = self.table.reindex(map_unique(smiles, _key).to_numpy())
        out.index = smiles.to_numpy()
        return out

    def transform_pca(self, smiles: Iterable[str], variance: float = 0.9, prefix: str = "cm_pc") -> pd.DataFrame:
        """Embeddings of a molecule set reduced by PCA fitted on that set (unsupervised, no yields).

        Keeps the fewest components explaining ``variance`` of the set's embedding variance
        (at most n_molecules - 1), so 2048-d vectors become usable with tens of data points.
        """
        E = self.transform(smiles)
        missing = E.index[E.isna().any(axis=1)].tolist()
        if missing:
            raise ValueError(f"no CheMeleon embedding for {len(missing)} molecules, e.g. {missing[0]!r}")
        E = E.loc[:, E.std() > 0]
        pca = PCA(n_components=min(len(E) - 1, E.shape[1]), random_state=0).fit(E)
        k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), variance) + 1)
        Z = pca.transform(E)[:, :k]
        return pd.DataFrame(Z, index=E.index, columns=[f"{prefix}{i + 1}" for i in range(k)])
