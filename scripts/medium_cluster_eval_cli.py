import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import argparse


def safe_metrics(X, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return float("nan"), float("nan")
    return float(silhouette_score(X, labels)), float(davies_bouldin_score(X, labels))


def run_kmeans(X, k_min=2, k_max=12, seed=42):
    best = None
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        lab = km.fit_predict(X)
        sil, db = safe_metrics(X, lab)
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        if best is None or (np.isfinite(sil) and sil > best["silhouette"]):
            best = {"k": k, "labels": lab, "silhouette": sil, "davies_bouldin": db}
    return best, pd.DataFrame(rows)


def run_agglom(X, k_min=2, k_max=12, linkage="ward"):
    best = None
    rows = []
    for k in range(k_min, k_max + 1):
        ac = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        lab = ac.fit_predict(X)
        sil, db = safe_metrics(X, lab)
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        if best is None or (np.isfinite(sil) and sil > best["silhouette"]):
            best = {"k": k, "labels": lab, "silhouette": sil, "davies_bouldin": db}
    return best, pd.DataFrame(rows)


def run_dbscan_grid(X, eps_list=(0.5, 0.8, 1.0, 1.2, 1.5), min_samples_list=(5, 10)):
    best = None
    rows = []
    for eps in eps_list:
        for ms in min_samples_list:
            dbs = DBSCAN(eps=eps, min_samples=ms)
            lab = dbs.fit_predict(X)

            mask = lab != -1
            if mask.sum() < 10 or len(np.unique(lab[mask])) < 2:
                sil = float("nan")
                db = float("nan")
            else:
                sil, db = safe_metrics(X[mask], lab[mask])

            rows.append({
                "eps": eps, "min_samples": ms,
                "silhouette": sil, "davies_bouldin": db,
                "n_clusters": int(len(set(lab)) - (1 if -1 in lab else 0)),
                "n_noise": int((lab == -1).sum())
            })

            if best is None or (np.isfinite(sil) and sil > best["silhouette"]):
                best = {"eps": eps, "min_samples": ms, "labels": lab, "silhouette": sil, "davies_bouldin": db}

    return best, pd.DataFrame(rows)


def evaluate_one(name, X, df, out_dir, label_col=None, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    Xs = StandardScaler().fit_transform(X)

    ari_labels = None
    ari_ok = False
    if label_col and label_col in df.columns and len(np.unique(df[label_col].astype(str).values)) > 1:
        ari_labels = df[label_col].astype(str).values
        ari_ok = True

    results = []

    best, curve = run_kmeans(Xs, seed=seed)
    curve.to_csv(os.path.join(out_dir, f"{name}_kmeans_curve.csv"), index=False)
    ari = float(adjusted_rand_score(ari_labels, best["labels"])) if ari_ok else float("nan")
    results.append({"repr": name, "algo": "kmeans", "param": f"k={best['k']}",
                    "silhouette": best["silhouette"], "davies_bouldin": best["davies_bouldin"], "ARI": ari})

    best, curve = run_agglom(Xs, linkage="ward")
    curve.to_csv(os.path.join(out_dir, f"{name}_agglom_curve.csv"), index=False)
    ari = float(adjusted_rand_score(ari_labels, best["labels"])) if ari_ok else float("nan")
    results.append({"repr": name, "algo": "agglomerative", "param": f"k={best['k']},linkage=ward",
                    "silhouette": best["silhouette"], "davies_bouldin": best["davies_bouldin"], "ARI": ari})

    best, grid = run_dbscan_grid(Xs)
    grid.to_csv(os.path.join(out_dir, f"{name}_dbscan_grid.csv"), index=False)
    ari = float(adjusted_rand_score(ari_labels, best["labels"])) if ari_ok else float("nan")
    results.append({"repr": name, "algo": "dbscan", "param": f"eps={best['eps']},min_samples={best['min_samples']}",
                    "silhouette": best["silhouette"], "davies_bouldin": best["davies_bouldin"], "ARI": ari})

    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--label_col", default="")
    ap.add_argument("--use_hybrid", action="store_true")
    args = ap.parse_args()

    data_dir = args.data_dir
    df = pd.read_csv(os.path.join(data_dir, "dataset_clean.csv"))

    reps = {}
    reps["audio_features"] = np.load(os.path.join(data_dir, "audio_features_clean.npy"))

    if args.use_hybrid:
        reps["hybrid_audio_lyrics"] = np.load(os.path.join(data_dir, "results_hybrid", "hybrid_features.npy"))

    out_dir = os.path.join(data_dir, "results_medium_clustering")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for name, X in reps.items():
        print("Evaluating:", name, "shape:", X.shape)
        r = evaluate_one(name, X, df, os.path.join(out_dir, name),
                         label_col=args.label_col if args.label_col else None, seed=42)
        all_rows.append(r)

    summary = pd.concat(all_rows, ignore_index=True)
    summary_path = os.path.join(out_dir, "medium_clustering_metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved summary:", summary_path)
    print(summary)

if __name__ == "__main__":
    main()
