import pandas as pd
import numpy as np

def select_by_corr_clusters(df, id_cols=None, threshold=0.70, cross_only=False):
    if id_cols is None:
        id_cols = {'SMILES','ID'}
    feats = [c for c in df.columns if c not in set(id_cols)]
    X = df[feats].select_dtypes(include='number')
    kr = [c for c in X.columns if c.endswith('_kr')]
    nonkr = [c for c in X.columns if not c.endswith('_kr')]
    corr_abs = X.corr().abs()
    mask = (corr_abs >= threshold) & ~np.eye(len(corr_abs), dtype=bool)
    if cross_only:
        same_group = pd.DataFrame(False, index=corr_abs.index, columns=corr_abs.columns)
        same_group.loc[kr, kr] = True
        same_group.loc[nonkr, nonkr] = True
        mask = mask & ~same_group
    adj = {c: set(corr_abs.columns[mask.loc[c]]) for c in corr_abs.index}
    visited = set()
    components = []
    for node in corr_abs.index:
        if node in visited:
            continue
        stack = [node]
        comp = set()
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            stack.extend(list(adj[n] - visited))
        components.append(sorted(comp))
    variances = X.var()
    kept = []
    dropped = []
    clusters = []
    for comp in components:
        if len(comp) == 1:
            kept.append(comp[0])
            clusters.append(comp)
            continue
        scores = {}
        for f in comp:
            others = [o for o in comp if o != f]
            scores[f] = corr_abs.loc[f, others].mean(skipna=True)
        best = sorted(comp, key=lambda f: (scores[f], -variances.get(f, 0.0), f))[0]
        kept.append(best)
        dropped.extend([f for f in comp if f != best])
        clusters.append(comp)
    kept = sorted(kept)
    dropped = sorted(set(dropped))
    return kept, dropped, clusters, corr_abs

df = pd.read_csv("C:/Users/Lyubomir/Downloads/fey-kraken-reduced.csv")
kept, dropped, clusters, corr = select_by_corr_clusters(df, id_cols=['SMILES','No.'], threshold=0.70)
kept_cross, dropped_cross, clusters_cross, corr = select_by_corr_clusters(df, id_cols=['SMILES','No.'], threshold=0.70, cross_only=True)

print(dropped)
print(len(dropped))
print(len(kept))
print(kept)
df_reduced = df.drop(columns=dropped, errors='ignore')
df_reduced.to_csv('C:/Users/Lyubomir/Downloads/fey-kraken-reduced-corr.csv', index=False)
