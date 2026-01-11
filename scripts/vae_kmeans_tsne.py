import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE

DATA_CSV = "data/processed/dataset_clean.csv"
LATENT_NPY = "data/processed/results_vae/latent_mu.npy"
OUT_DIR = "data/processed/results_vae"

def main(k_min=2, k_max=12, seed=42):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_CSV)
    Z = np.load(LATENT_NPY).astype(np.float32)

    # --- Find best k ---
    rows = []
    best_k, best_s = None, -1

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = km.fit_predict(Z)
        sil = silhouette_score(Z, labels)
        ch = calinski_harabasz_score(Z, labels)
        rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch})
        if sil > best_s:
            best_s, best_k = sil, k

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT_DIR, "vae_latent_kmeans_metrics.csv"), index=False)
    print(f"Best k by silhouette: {best_k} (silhouette={best_s:.4f})")

    # --- Final clustering with best k ---
    km = KMeans(n_clusters=best_k, n_init=20, random_state=seed)
    labels = km.fit_predict(Z)

    out = df.copy()
    out["cluster"] = labels
    out.to_csv(os.path.join(OUT_DIR, "vae_cluster_assignments.csv"), index=False)

    pd.crosstab(out["cluster"], out["language"]).to_csv(
        os.path.join(OUT_DIR, "vae_cluster_language_crosstab.csv")
    )

    # --- t-SNE visualization ---
    perplexity = min(30, (len(Z) - 1) // 3)
    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity)
    Y = tsne.fit_transform(Z)

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=12)
    plt.title(f"t-SNE of VAE latent (colored by cluster, k={best_k})")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "tsne_latent_by_cluster.png"), dpi=150)
    plt.close()

    is_bn = (out["language"].astype(str).values == "bn").astype(int)
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=is_bn, s=12)
    plt.title("t-SNE of VAE latent (colored by language: bn=1, en=0)")
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "tsne_latent_by_language.png"), dpi=150)
    plt.close()

    print("Saved t-SNE plots and VAE clustering outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
