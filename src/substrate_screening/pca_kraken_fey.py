import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

df = pd.read_csv("C:/Users/Lyubomir/Downloads/fey-kraken-reduced-corr.csv", usecols=lambda c: not str(c).startswith("Unnamed"))
out_dir = r"C:\Users\Lyubomir\Downloads"
os.makedirs(out_dir, exist_ok=True)

def pca_on_descriptors(df, exclude=None, n_components=2):
    exclude = set(exclude or [])
    desc_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[desc_cols].to_numpy()
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0).fit(X)
    scores = pca.transform(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    scores_df = pd.DataFrame(scores, index=df.index, columns=[f"PC{i+1}" for i in range(n_components)])
    loadings_df = pd.DataFrame(loadings, index=desc_cols, columns=[f"PC{i+1}" for i in range(n_components)])
    return pca, scores_df, loadings_df

def save_pca_plots(df, exclude, out_dir, n_components=2):
    os.makedirs(out_dir, exist_ok=True)
    pca, scores_df, loadings_df = pca_on_descriptors(df, exclude=exclude, n_components=n_components)
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    is_kr = loadings_df.index.to_series().str.endswith("_kr")

    plt.figure()
    plt.scatter(loadings_df.loc[~is_kr, "PC1"], loadings_df.loc[~is_kr, "PC2"], label="fey")
    plt.scatter(loadings_df.loc[is_kr, "PC1"], loadings_df.loc[is_kr, "PC2"], label="kraken")
    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel(f"PC1 ({pc1_var:.1f}%)")
    plt.ylabel(f"PC2 ({pc2_var:.1f}%)")
    plt.title("PCA feature loadings")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.savefig(os.path.join(out_dir, "pca_loadings_reduced_corr.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(scores_df["PC1"], scores_df["PC2"], label="samples")
    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel(f"PC1 ({pc1_var:.1f}%)")
    plt.ylabel(f"PC2 ({pc2_var:.1f}%)")
    plt.title("PCA sample scores")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.savefig(os.path.join(out_dir, "pca_scores_reduced_corr.png"), dpi=300, bbox_inches="tight")
    plt.close()

save_pca_plots(df, exclude=["SMILES","No."], out_dir=r"C:\Users\Lyubomir\Downloads")
