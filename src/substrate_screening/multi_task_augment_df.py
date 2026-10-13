import pandas as pd
from rdkit import Chem

def canonicalize_smiles(df, col="SMILES", new_col="SMILES_canonical", isomeric=False):
    canon = []
    for s in df[col].astype(str):
        m = Chem.MolFromSmiles(s)
        canon.append(Chem.MolToSmiles(m, isomericSmiles=isomeric) if m is not None else None)
    out = df.copy()
    out[new_col] = canon
    out = out.dropna(subset=[new_col])
    return out

def augment_df2_with_df1(df1, df2, key_col="SMILES", exclude=None, isomeric=False, suffix="_kr"):
    exclude = set(exclude or [])
    c1 = canonicalize_smiles(df1, col=key_col, new_col=f"{key_col}_canonical", isomeric=isomeric)
    c2 = canonicalize_smiles(df2, col=key_col, new_col=f"{key_col}_canonical", isomeric=isomeric)
    c1u = c1.drop_duplicates(subset=[f"{key_col}_canonical"])
    cols_to_transfer = [c for c in c1u.columns if c not in exclude and c not in {key_col, f"{key_col}_canonical"}]
    c1_sub = c1u[[f"{key_col}_canonical"] + cols_to_transfer].copy()
    c1_sub.columns = [f"{key_col}_canonical"] + [f"{c}{suffix}" for c in cols_to_transfer]
    merged = c2.merge(c1_sub, on=f"{key_col}_canonical", how="left")
    merged = merged.drop(columns=[f"{key_col}_canonical"])
    return merged

df1 = pd.read_csv("C:/Users/Lyubomir/Downloads/kraken.csv", usecols=lambda c: not str(c).startswith("Unnamed"))
df2 = pd.read_csv("../data/ligands/fey_monop.csv")

exclude_cols = ["ID", "SMILES"]
df2_augmented = augment_df2_with_df1(df1, df2, key_col="SMILES", exclude=exclude_cols, isomeric=False, suffix="_kr")
df2_augmented.dropna(subset=["fmo_e_lumo_boltz_kr"], inplace=True)
df2_augmented = df2_augmented.loc[:, ~df2_augmented.columns.str.contains("Unnamed")]


df2_augmented.to_csv("C:/Users/Lyubomir/Downloads/fey-kraken.csv", index=False)


def canonical_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(m, isomericSmiles=False)