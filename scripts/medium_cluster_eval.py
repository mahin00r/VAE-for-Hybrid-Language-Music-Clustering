import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score


def safe_metrics(X, labels):
    """Return (silhouette, davies_bouldin) or (nan,nan) if invalid."""
    uniq = np.unique(labels)
    # Need at least 2 clusters and not all points in separate clusters
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

            # Ignore noise for silhouette/DB if present
            mask = lab != -1
            if mask.sum() < 10 or len(np.unique(lab[mask])) < 2:
                sil = float("nan")
                db = float("nan")
            else:
                sil, db = safe_metrics(X[mask], lab[mask])

            rows.append({"eps": eps, "min_samples": ms, "silhouette": sil, "davies_bouldin": db,
                         "n_clusters": int(len(set(lab)) - (1 if -1 in lab else 0)),
                         "n_noise": int((lab == -1).sum())})

            if best is None or (np.isfinite(sil) and sil > best["silhouette"]):
                best = {"eps": eps, "min_samples": ms, "labels": lab, "silhouette": sil, "davies_bouldin": db}

    return best, pd.DataFrame(rows)


def evaluate_representation(name, X, df, out_dir, label_col="language", seed=42):
    os.makedirs(out_dir, exist_ok=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ARI (only if label_col exists and has >1 class)
    ari_labels = None
    ari_ok = False
    if label_col in df.columns:
        y = df[label_col].astype(str).values
        if len(np.unique(y)) > 1:
            ari_labels = y
            ari_ok = True

    results = []

    # KMeans
    best_km, km_curve = run_kmeans(Xs, seed=seed)
    km_curve.to_csv(os.path.join(out_dir, f"{name}_kmeans_curve.csv"), index=False)
    lab = best_km["labels"]
    ari = float(adjusted_rand_score(ari_labels, lab)) if ari_ok else float("nan")
    results.append({"repr": name, "algo": "kmeans", "param": f"k={best_km['k']}",
                    "silhouette": best_km["silhouette"], "davies_bouldin": best_km["davies_bouldin"], "ARI": ari})

    out = df.copy()
    out["cluster"] = lab
    out.to_csv(os.path.join(out_dir, f"{name}_kmeans_assignments.csv"), index=False)
    if "language" in df.columns:
        pd.crosstab(out["cluster"], out["language"]).to_csv(os.path.join(out_dir, f"{name}_kmeans_crosstab_language.csv"))

    # Agglomerative
    best_ag, ag_curve = run_agglom(Xs, linkage="ward")
    ag_curve.to_csv(os.path.join(out_dir, f"{name}_agglom_curve.csv"), index=False)
    lab = best_ag["labels"]
    ari = float(adjusted_rand_score(ari_labels, lab)) if ari_ok else float("nan")
    results.append({"repr": name, "algo": "agglomerative", "param": f"k={best_ag['k']},linkage=ward",
                    "silhouette": best_ag["silhouette"], "davies_bouldin": best_ag["davies_bouldin"], "ARI": ari})

    out = df.copy()
    out["cluster"] = lab
    out.to_csv(os.path.join(out_dir, f"{name}_agglom_assignments.csv"), index=False)

    # DBSCAN
    best_db, db_curve = run_dbscan_grid(Xs)
    db_curve.to_csv(os.path.join(out_dir, f"{name}_dbscan_grid.csv"), index=False)
    lab = best_db["labels"]
    ari = float(adjusted_rand_score(ari_labels, lab)) if ari_ok else float("nan")
    results.append({"repr": name, "algo": "dbscan", "param": f"eps={best_db['eps']},min_samples={best_db['min_samples']}",
                    "silhouette": best_db["silhouette"], "davies_bouldin": best_db["davies_bouldin"], "ARI": ari})

    out = df.copy()
    out["cluster"] = lab
    out.to_csv(os.path.join(out_dir, f"{name}_dbscan_assignments.csv"), index=False)

    return pd.DataFrame(results)


def main():
    data_dir = "data/processed_medium"
    df = pd.read_csv(os.path.join(data_dir, "dataset_clean.csv"))

    # Representations we will evaluate:
    reps = {}

    # 1) MFCC/chroma vector features (baseline representation)
    af = np.load(os.path.join(data_dir, "audio_features_clean.npy"))
    reps["audio_features"] = af

    # 2) ConvVAE latent (after you train conv VAE)
    conv_latent_path = os.path.join(data_dir, "results_conv_vae", "latent_mu.npy")
    if os.path.exists(conv_latent_path):
        reps["convvae_latent"] = np.load(conv_latent_path)

    # 3) hybrid features (optional; created later)
    hybrid_path = os.path.join(data_dir, "results_hybrid", "hybrid_features.npy")
    if os.path.exists(hybrid_path):
        reps["hybrid_audio_lyrics"] = np.load(hybrid_path)

    out_dir = os.path.join(data_dir, "results_medium_clustering")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for name, X in reps.items():
        print("Evaluating:", name, "X shape:", X.shape)
        rep_dir = os.path.join(out_dir, name)
        r = evaluate_representation(name, X, df, rep_dir, label_col="language", seed=42)
        all_rows.append(r)

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "medium_clustering_metrics_summary.csv"), index=False)
    print("Saved summary:", os.path.join(out_dir, "medium_clustering_metrics_summary.csv"))
    print(summary)


if __name__ == "__main__":
    main()
