import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ligands_120 = pd.read_csv('./data/ligands/stock_commercial_kraken_match.csv')
ks_60 = pd.read_csv('./data/ligands/ks60.csv')

unexplored_60 = ligands_120[~ligands_120["SMILES"].isin(ks_60["SMILES"])].copy()
unexplored_60.reset_index(drop=True, inplace=True)

poisonous_25 = pd.read_csv("./data/ligands/poisonous.csv")

FORBIDDEN_EXACT = {"ks_rank", "No.", "SMILES", "CAS", "location"}

def _descriptor_cols(df, other_df=None):
    forbidden_lower = {c.lower() for c in FORBIDDEN_EXACT}
    keep = [c for c in df.columns
            if c.lower() not in forbidden_lower
            and "unnamed" not in c.lower()]
    num = df[keep].select_dtypes(include=[np.number]).columns.tolist()
    if other_df is not None:
        other_num = _descriptor_cols(other_df)
        num = sorted(set(num).intersection(other_num))
    return num

def find_neighbors(poisonous_25: pd.DataFrame,
                   unexplored_60: pd.DataFrame,
                   n_neighbors: int = 3) -> pd.DataFrame:
    desc_cols = _descriptor_cols(poisonous_25, unexplored_60)
    if not desc_cols:
        raise ValueError("No usable descriptor columns found in common between the two dataframes.")

    ref = unexplored_60.copy()
    qry = poisonous_25.copy()

    ref_desc = ref[desc_cols].replace([np.inf, -np.inf], np.nan)
    qry_desc = qry[desc_cols].replace([np.inf, -np.inf], np.nan)

    medians = ref_desc.median(numeric_only=True)
    ref_desc = ref_desc.fillna(medians)
    qry_desc = qry_desc.fillna(medians)

    scaler = StandardScaler()
    X_ref = scaler.fit_transform(ref_desc.values)
    X_qry = scaler.transform(qry_desc.values)

    k = min(n_neighbors, len(ref))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_ref)
    dists, idxs = nn.kneighbors(X_qry, n_neighbors=k, return_distance=True)

    out_rows = []
    role_order_map = {"ligand": 0, "neighbour1": 1, "neighbour2": 2, "neighbour3": 3}

    for i, (row_idx, row) in enumerate(qry.iterrows()):
        group_id = f"ligand_{i+1}"
        # ligand row
        lig = row.copy()
        lig["_group"] = group_id
        lig["_group_idx"] = i + 1
        lig["_role"] = "ligand"
        lig["_role_order"] = role_order_map["ligand"]
        lig["_distance"] = np.nan
        out_rows.append(lig)

        # neighbour rows
        for j, (ref_i, dist) in enumerate(zip(idxs[i], dists[i]), start=1):
            nb = ref.iloc[ref_i].copy()
            nb["_group"] = group_id
            nb["_group_idx"] = i + 1
            nb["_role"] = f"neighbour{j}"
            nb["_role_order"] = role_order_map.get(nb["_role"], 99)
            nb["_distance"] = float(dist)
            out_rows.append(nb)

    out = pd.DataFrame(out_rows)

    # Proper ordering: numeric group index, then ligand/neighbours
    out = out.sort_values(["_group_idx", "_role_order"]).reset_index(drop=True)

    # Drop helper sort column(s) but keep _group/_role/_distance
    out = out.drop(columns=["_role_order", "_group_idx"])

    # Drop Unnamed columns from the final output
    out = out[[c for c in out.columns if "unnamed" not in c.lower()]]

    # (Optional) remember which descriptors were used
    out.attrs["descriptor_columns_used"] = desc_cols
    return out


result_df = find_neighbors(poisonous_25, unexplored_60, n_neighbors=3)
result_df.to_csv("./data/ligands/nearest_neighbours.csv", index=False)

