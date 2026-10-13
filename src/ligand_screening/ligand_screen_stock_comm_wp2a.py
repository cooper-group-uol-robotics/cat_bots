"""Legacy WP2A step: match the lab's ligand stock against Kraken and commercial availability.

The inputs ``phosphine_stock.csv`` and ``pr3_stock.csv`` are internal chemical-inventory exports and are
not distributed with this repository, so this script cannot be run as-is. Its outputs
(``stock_kraken_match.csv``, ``stock_commercial_kraken_match.csv``) are included without storage locations.
"""

import re
import time
from functools import lru_cache

import numpy as np
import pubchempy as pcp
from rdkit import Chem
import pandas as pd

def _clean_cas(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    if re.fullmatch(r"\d{5,7}", s):
        core, check = s[:-1], s[-1]
        mid = core[-2:]
        left = core[:-2]
        s = f"{left}-{mid}-{check}"
    return s

@lru_cache(maxsize=None)
def cas_to_smiles(cas_raw, pause=0.2):
    cas = _clean_cas(cas_raw)
    if not cas:
        return None

    comps = []
    try:
        comps = pcp.get_compounds(cas, "rn")
    except Exception:
        comps = []

    if not comps:
        syn_variants = (cas, f"CAS-{cas}", f"CAS {cas}", f"CAS No. {cas}")
        for term in syn_variants:
            try:
                comps = pcp.get_compounds(term, "synonym")
                if comps:
                    break
            except Exception:
                pass

    if not comps:
        for term in (cas, cas.replace("-", "")):
            try:
                _c = pcp.get_compounds(term, "name")
                if _c:
                    comps = _c
                    break
            except Exception:
                pass

    if pause:
        time.sleep(pause)

    if not comps:
        return None

    cmpd = comps[0]
    smi = getattr(cmpd, "canonical_smiles", None) or getattr(cmpd, "isomeric_smiles", None)
    print(f"SMILES: {smi}")
    return smi

def rdkit_canonicalize(smiles):
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    s = str(smiles).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles= False, canonical=True)


phosphine_stock = pd.read_csv("../data/ligands/phosphine_stock.csv")
pr3_stock = pd.read_csv("../data/ligands/pr3_stock.csv")
commercial_ligs = pd.read_csv("../data/ligands/commercial_ligs.csv")

kraken = pd.read_csv("../data/ligands/kraken_reduced_0.7_no_delta.csv")


all_cas = pd.concat([
    phosphine_stock["CAS No."],
    pr3_stock["CAS No."]
], ignore_index=True)

cas_unique = (all_cas.dropna()
                    .astype(str)
                    .map(_clean_cas)
                    .dropna()
                    .unique())

cas_to_smi_map = {cas: cas_to_smiles(cas) for cas in cas_unique}

phosphine_stock["SMILES"] = (
    phosphine_stock["CAS No."].astype(str).map(_clean_cas).map(cas_to_smi_map).replace({None: np.nan})
)
pr3_stock["SMILES"] = (
    pr3_stock["CAS No."].astype(str).map(_clean_cas).map(cas_to_smi_map).replace({None: np.nan})
)

commercial_ligs["SMILES"] = commercial_ligs["Canonical SMILES"].apply(rdkit_canonicalize)

kraken["SMILES"] = kraken["SMILES"].apply(rdkit_canonicalize)

for df in (phosphine_stock, pr3_stock, commercial_ligs, kraken):
    if "_SMIKEY" not in df.columns:
        df["_SMIKEY"] = df["SMILES"].apply(rdkit_canonicalize)

def find_col(df, candidates):
    cand_norm = [c.strip().lower() for c in candidates]
    for col in df.columns:
        if col.strip().lower() in cand_norm:
            return col
    return None

PHOS_CAS_COL = find_col(phosphine_stock, ["CAS No.", "CAS no.", "CAS"])
PHOS_LOC_COL = find_col(phosphine_stock, ["Storage location", "Location"])
PR3_CAS_COL  = find_col(pr3_stock,        ["CAS No.", "CAS no.", "CAS"])
PR3_LOC_COL  = find_col(pr3_stock,        ["Storage location", "Location"])
COMM_CAS_COL = find_col(commercial_ligs,  ["CAS Registry Number", "CAS No.", "CAS"])

def build_maps(df, cas_col, loc_col=None):
    if cas_col is None:
        cas_map = {}
    else:
        cas_map = (df[["_SMIKEY", cas_col]]
                   .dropna(subset=["_SMIKEY"])
                   .drop_duplicates("_SMIKEY", keep="first")
                   .set_index("_SMIKEY")[cas_col]
                   .astype(str)
                   .str.strip()
                   .replace({"nan": np.nan})
                   .to_dict())
    if loc_col is None:
        loc_map = {}
    else:
        loc_map = (df[["_SMIKEY", loc_col]]
                   .dropna(subset=["_SMIKEY"])
                   .drop_duplicates("_SMIKEY", keep="first")
                   .set_index("_SMIKEY")[loc_col]
                   .to_dict())
    return cas_map, loc_map

phos_cas_map, phos_loc_map = build_maps(phosphine_stock, PHOS_CAS_COL, PHOS_LOC_COL)
pr3_cas_map,  pr3_loc_map  = build_maps(pr3_stock,        PR3_CAS_COL,  PR3_LOC_COL)
comm_cas_map, _            = build_maps(commercial_ligs,  COMM_CAS_COL, None)

kraken = kraken.copy()
kraken["_SMIKEY"] = kraken["_SMIKEY"]

kraken["match_phosphine"]  = kraken["_SMIKEY"].isin(phosphine_stock["_SMIKEY"].dropna().unique())
kraken["match_pr3"]        = kraken["_SMIKEY"].isin(pr3_stock["_SMIKEY"].dropna().unique())
kraken["match_commercial"] = kraken["_SMIKEY"].isin(commercial_ligs["_SMIKEY"].dropna().unique())

kraken["CAS_phos"] = kraken["_SMIKEY"].map(phos_cas_map)
kraken["CAS_pr3"]  = kraken["_SMIKEY"].map(pr3_cas_map)
kraken["CAS_comm"] = kraken["_SMIKEY"].map(comm_cas_map)

kraken["loc_phos"] = kraken["_SMIKEY"].map(phos_loc_map)
kraken["loc_pr3"]  = kraken["_SMIKEY"].map(pr3_loc_map)

def coalesce3(a, b, c):
    return a.fillna(b).fillna(c)

def coalesce2(a, b):
    return a.fillna(b)

mask_stock = kraken["match_phosphine"] | kraken["match_pr3"]
stock_kraken_match = kraken.loc[mask_stock].copy()
stock_kraken_match["CAS"]      = coalesce2(stock_kraken_match["CAS_phos"], stock_kraken_match["CAS_pr3"])
stock_kraken_match["location"] = coalesce2(stock_kraken_match["loc_phos"], stock_kraken_match["loc_pr3"])

stock_kraken_match.drop(
    columns=[
        "_SMIKEY","match_phosphine","match_pr3","match_commercial",
        "CAS_phos","CAS_pr3","CAS_comm","loc_phos","loc_pr3"
    ],
    inplace=True, errors="ignore"
)

mask_any = mask_stock | kraken["match_commercial"]
stock_commercial_kraken_match = kraken.loc[mask_any].copy()

stock_commercial_kraken_match["CAS"] = coalesce3(
    stock_commercial_kraken_match["CAS_phos"],
    stock_commercial_kraken_match["CAS_pr3"],
    stock_commercial_kraken_match["CAS_comm"],
)

stock_commercial_kraken_match["location"] = coalesce2(
    stock_commercial_kraken_match["loc_phos"],
    stock_commercial_kraken_match["loc_pr3"],
)

stock_commercial_kraken_match.drop(
    columns=[
        "_SMIKEY","match_phosphine","match_pr3","match_commercial",
        "CAS_phos","CAS_pr3","CAS_comm","loc_phos","loc_pr3"
    ],
    inplace=True, errors="ignore"
)

stock_kraken_match.to_csv("../data/ligands/stock_kraken_match.csv", index=False)
stock_commercial_kraken_match.to_csv("../data/ligands/stock_commercial_kraken_match.csv", index=False)