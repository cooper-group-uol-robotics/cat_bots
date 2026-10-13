import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.exceptions import NotFittedError

def kennard_stone_indices(X: np.ndarray, k: int) -> list[int]:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k={k} cannot exceed n_samples={n}")

    # Pairwise squared Euclidean distances
    G  = X @ X.T
    H  = np.sum(X**2, axis=1, keepdims=True)
    D2 = H + H.T - 2 * G
    np.fill_diagonal(D2, 0.0)

    # farthest pair
    i1, i2 = np.unravel_index(np.argmax(D2), D2.shape)
    selected = [int(i1), int(i2)]

    # greedy max–min
    min_dist2 = np.minimum(D2[:, i1], D2[:, i2])
    min_dist2[i1] = -np.inf
    min_dist2[i2] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(min_dist2))
        selected.append(nxt)
        min_dist2 = np.minimum(min_dist2, D2[:, nxt])
        min_dist2[nxt] = -np.inf

    return selected

def canonicalize_smiles(s: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def morgan_fp_bits(mol, radius=2, nBits=2048) -> np.ndarray:
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def bits_to_string(bits: np.ndarray) -> str:
    return "".join("1" if b else "0" for b in bits.astype(bool))

def ks_select_substrates_by_fingerprint(
    df: pd.DataFrame,
    k: int = 36,
    smiles_col_candidates: tuple[str, ...] = ("Canonical SMILES", "SMILES", "smiles"),
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> tuple[pd.DataFrame, list[int], list[str]]:

    df = df.copy()

    smiles_col = next((c for c in smiles_col_candidates if c in df.columns), None)
    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Tried: {smiles_col_candidates}")

    canon = df[smiles_col].astype(str).map(canonicalize_smiles)
    mask_valid = canon.notna()
    if mask_valid.sum() < k:
        raise ValueError(
            f"Only {int(mask_valid.sum())} valid SMILES after canonicalization; k={k} is too large."
        )

    df = df.loc[mask_valid].copy()
    df["canonical_smiles"] = canon[mask_valid].values

    # Compute fingerprints
    fps = []
    for s in df["canonical_smiles"]:
        mol = Chem.MolFromSmiles(s)
        fp  = morgan_fp_bits(mol, radius=fp_radius, nBits=fp_bits)
        fps.append(fp)
    X = np.vstack(fps)  # shape: (n_rows_valid, fp_bits)
    # KS selection
    order = kennard_stone_indices(X, k)
    # Add human-readable fingerprint as a bitstring column for traceability
    df["morgan_fp_bits"] = [bits_to_string(fp) for fp in fps]
    # Selected rows in KS order + rank
    df_sel = df.iloc[order].copy()
    df_sel.insert(0, "ks_rank", range(1, len(df_sel) + 1))
    return df_sel, order, ["morgan_fp_bits"]

if __name__ == "__main__":
    df = pd.read_csv(
        "../data/substrates/completed_orca_jobs_culled_abs_minmax.csv",
        usecols=lambda c: not str(c).startswith("Unnamed"),
    )
    df = df.dropna(axis=1, how="all")

    df_36, ks_order, used_features = ks_select_substrates_by_fingerprint(
        df, k=36, smiles_col_candidates=("Canonical SMILES", "SMILES", "smiles"),
        fp_radius=2, fp_bits=2048
    )

    out_path = "../data/substrates/ks36_culled_abs_minmax_morgan.csv"
    df_36.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Feature used:", used_features[0])
    print("First 5 KS indices:", ks_order[:5])