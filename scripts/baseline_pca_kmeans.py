import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Baseline: Standardize -> PCA -> KMeans, pick best k by silhouette.")
    ap.add_argument("--data_dir", default="data/processed", help="Folder containing dataset_clean.csv and audio_features_clean.npy")
    ap.add_argument("--csv", default=None, help="Optional explicit csv path (defaults to data_dir/dataset_clean.csv)")
    ap.add_argument("--features", default=None, help="Optional explicit npy path (defaults to data_dir/audio_features_clean.npy)")
    ap.add_argument("--out_dir", default=None, help="Output folder (defaults to data_dir/results_baseline)")
    ap.add_argument("--pca_dim", type=int, default=16, help="PCA output dimension")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = Path(args.csv) if args.csv else (data_dir / "dataset_clean.csv")
    feat_path = Path(args.features) if args.features else (data_dir / "audio_features_clean.npy")
    out_dir = Path(args.out_dir) if args.out_dir else (data_dir / "results_baseline")

    ensure_dir(out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}")

    df = pd.read_csv(csv_path)
    X = np.load(feat_path)

    if len(df) != X.shape[0]:
        raise RuntimeError(f"Row mismatch: csv rows={len(df)} but features rows={X.shape[0]}")

    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(args.pca_dim, Xs.shape[1]), random_state=args.seed)
    Z = pca.fit_transform(Xs)

    rows = []
    best_k = None
    best_sil = -1.0

    for k in range(args.k_min, args.k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
        labels = km.fit_predict(Z)

        # Silhouette requires at least 2 clusters and less than n_samples clusters
        sil = silhouette_score(Z, labels)
        ch = calinski_harabasz_score(Z, labels)

        rows.append({"k": k, "silhouette": float(sil), "calinski_harabasz": float(ch)})

        if sil > best_sil:
            best_sil = sil
            best_k = k

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "baseline_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # Fit final best model
    km = KMeans(n_clusters=best_k, n_init=20, random_state=args.seed)
    labels = km.fit_predict(Z)

    out = df.copy()
    out["cluster"] = labels
    assignments_path = out_dir / "cluster_assignments.csv"
    out.to_csv(assignments_path, index=False)

    # Crosstab vs language if present
    if "language" in out.columns:
        crosstab_path = out_dir / "cluster_language_crosstab.csv"
        pd.crosstab(out["cluster"], out["language"]).to_csv(crosstab_path)
    else:
        crosstab_path = None

    print(f"Saved metrics: {metrics_path}")
    print(f"Best k by silhouette: {best_k} (silhouette={best_sil:.4f})")
    print(f"Saved assignments: {assignments_path}")
    if crosstab_path:
        print(f"Saved cluster-language table: {crosstab_path}")


if __name__ == "__main__":
    main()
