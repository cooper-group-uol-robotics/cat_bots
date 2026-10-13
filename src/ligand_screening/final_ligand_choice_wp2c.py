import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

ligands_csv     = "../data/ligands/fey_monop.csv"
mt_bundle_path  = "../data/ML_multitask/models/multitask_enet_top10.joblib"
out_dir         = Path("../data/ML_multitask/preds")
out_smiles      = out_dir / "top20_ligands_by_avg_yield_smiles.csv"
out_full        = out_dir / "preds_all_ligands_per_substrate.csv"

out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ligands_csv)
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].copy()
if "SMILES" not in df.columns:
    raise ValueError("The ligands file must contain a 'SMILES' column.")
df["SMILES"] = df["SMILES"].astype(str).str.strip()
df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

# Drop non-descriptor columns you know about
id_cols = [c for c in ["No.", "SMILES", "Yield (%)"] if c in df.columns]
desc_df = df.drop(columns=id_cols, errors="ignore").copy()

bundle = load(mt_bundle_path)
pipe   = bundle["pipeline"]
tasks  = bundle["tasks"]

# Figure out which input columns the pipeline expects BEFORE imputation/scaling
if "select" in pipe.named_steps:
    # top-K model: ColumnTransformer selects these by name
    expected_cols = list(pipe.named_steps["select"].transformers_[0][2])
else:
    # all-features model: imputer was fit on these
    expected_cols = list(pipe.named_steps["imputer"].feature_names_in_)

# Align descriptors to expected columns (add missing as NaN → will be imputed)
X = desc_df.filter(items=expected_cols).reindex(columns=expected_cols)
X = X.apply(pd.to_numeric, errors="coerce")

# predict per substrate and average
Y_pred = pipe.predict(X)

preds_full = pd.DataFrame(Y_pred, columns=[str(t) for t in tasks])
preds_full.insert(0, "SMILES", df["SMILES"].values)
preds_full.to_csv(out_full, index=False)

# Average predicted yield across substrates
pred_avg = Y_pred.mean(axis=1)
top_idx = np.argsort(-pred_avg)[:20] #top 20

top20 = df.loc[top_idx, ["SMILES"]].reset_index(drop=True)
top20.to_csv(out_smiles, index=False)

print(f"Saved per-substrate predictions for all ligands to: {out_full}")
print(f"Saved top-20 ligands by average predicted yield to: {out_smiles}")
print("Top-20 SMILES:", ", ".join(top20['SMILES'].tolist()))
