import pandas as pd
from rdkit import Chem

kraken = pd.read_csv("../data/ligands/kraken.csv")
allphosphines = pd.read_csv("../data/ligands/AllPhosphines.csv")
def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

kraken["canonical_SMILES"] = kraken["SMILES"].apply(canonicalize_smiles)
#how many of those 400 are not inside the commercial

overlap = allphosphines[allphosphines["Canonical_SMILES"].isin(kraken["canonical_SMILES"])]
stock_commercial = pd.read_csv("stock_commercial_kraken_match.csv")
stock_commercial["canonical_SMILES"] = stock_commercial["SMILES"].apply(canonicalize_smiles)
overlap2 = overlap[~overlap["Canonical_SMILES"].isin(stock_commercial["canonical_SMILES"])]


print(overlap.info())
print(overlap2.info())

overlap2.to_csv("../data/ligands/ligand_expansion.csv", index=False)