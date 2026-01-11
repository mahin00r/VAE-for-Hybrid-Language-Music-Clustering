import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def save_scatter(Z2, labels, title, out_path):
    plt.figure()
    plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=12)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to .npy features (N,D)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=2, help="k for KMeans (used for coloring)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X = np.load(args.features).astype(np.float32)
    X = StandardScaler().fit_transform(X)

    # cluster labels (for coloring)
    km = KMeans(n_clusters=args.k, n_init=20, random_state=args.seed)
    labels = km.fit_predict(X)

    n = X.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))

    tsne = TSNE(n_components=2, random_state=args.seed, init="pca", perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    save_scatter(X_tsne, labels,
                 f"t-SNE (k={args.k})",
                 os.path.join(args.out_dir, "tsne_by_cluster.png"))

    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        X_umap = reducer.fit_transform(X)
        save_scatter(X_umap, labels,
                     f"UMAP (k={args.k})",
                     os.path.join(args.out_dir, "umap_by_cluster.png"))
    else:
        print("UMAP not installed. Install with: pip install umap-learn")

    print("Saved plots in:", args.out_dir)


if __name__ == "__main__":
    main()
