import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def kennard_stone_indices(X: np.ndarray, k: int) -> list[int]:

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k={k} cannot exceed n_samples={n}")

    # Pairwise squared Euclidean distances (no SciPy needed)
    G = X @ X.T                           # Gram matrix
    H = np.sum(X**2, axis=1, keepdims=True)
    D2 = H + H.T - 2 * G                  # squared distances
    np.fill_diagonal(D2, 0.0)

    # Seed with the farthest pair
    i1, i2 = np.unravel_index(np.argmax(D2), D2.shape)
    selected = [int(i1), int(i2)]

    # Greedy max–min: track each point's min distance to the selected set
    min_dist2 = np.minimum(D2[:, i1], D2[:, i2])
    min_dist2[i1] = -np.inf
    min_dist2[i2] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(min_dist2))
        selected.append(nxt)
        min_dist2 = np.minimum(min_dist2, D2[:, nxt])
        min_dist2[nxt] = -np.inf

    return selected

def ks_select_substrates(
    df: pd.DataFrame,
    k: int = 24,
    id_cols: tuple[str, str] = ("No.", "SMILES"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[str]]:

    # Build descriptor list: all columns except id_cols, keep only numeric
    candidates = df.drop(columns=list(id_cols), errors="ignore")
    features = candidates.select_dtypes(include=[np.number]).columns.tolist()
    if not features:
        raise ValueError("No numeric descriptor columns found after excluding id columns.")
    X = df[features].to_numpy()

    # Impute missing values (median) and standardize
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    # KS selection order
    order = kennard_stone_indices(X, k)

    # Slice dataframes in KS order and annotate with selection rank
    df_24 = df.iloc[order].copy()
    df_24.insert(0, "ks_rank", range(1, len(df_24) + 1))  # 1-based rank in selection order

    return df_24, order, features

ligands_60 = pd.read_csv("../data/ligands/ks60.csv")

df = pd.read_csv("../data/ligands/ks60.csv",
                 usecols=lambda c: not str(c).startswith("Unnamed"))

df = df.dropna(axis=1, how="all")
df_24, ks_order, used_features = ks_select_substrates(df, k=24)
df_24.to_csv("../data/substrates/ks24_wp2c.csv", index=False)
print("Features used:", used_features)



