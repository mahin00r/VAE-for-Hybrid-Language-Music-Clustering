import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

import matplotlib.pyplot as plt

try:
    import umap  # type: ignore
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

from sklearn.manifold import TSNE


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Purity = (1/n) sum_k max_j |c_k ∩ t_j|  (from your PDF) :contentReference[oaicite:4]{index=4}
    df = pd.DataFrame({"t": y_true, "c": y_pred})
    # treat noise (-1) as its own cluster too
    tab = pd.crosstab(df["c"], df["t"])
    return float(tab.max(axis=1).sum() / max(len(df), 1))


def safe_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    uniq = np.unique(labels)
    if len(uniq) < 2:
        out["silhouette"] = np.nan
        out["davies_bouldin"] = np.nan
        out["calinski_harabasz"] = np.nan
        return out

    out["silhouette"] = float(silhouette_score(X, labels))
    out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    return out


def label_metrics(labels_pred: np.ndarray, labels_true: np.ndarray) -> Dict[str, Any]:
    # NMI + ARI (from PDF metric list) 
    if len(np.unique(labels_true)) < 2:
        return {"nmi": np.nan, "ari": np.nan, "purity": np.nan}
    return {
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
        "purity": float(purity_score(labels_true, labels_pred)),
    }


def embed_2d(X: np.ndarray, seed: int) -> np.ndarray:
    if HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(X)
    # fallback
    return TSNE(n_components=2, random_state=seed, init="pca", perplexity=min(30, max(5, (len(X) - 1) // 3))).fit_transform(X)


def scatter_save(X2: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def categorical_to_int(series: pd.Series) -> Tuple[np.ndarray, Dict[int, str]]:
    cats = pd.Categorical(series.astype(str))
    codes = cats.codes.astype(int)
    mapping = {i: str(cat) for i, cat in enumerate(cats.categories)}
    return codes, mapping


def make_onehot(codes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(codes), n_classes), dtype=np.float32)
    out[np.arange(len(codes)), codes] = 1.0
    return out


def load_direct_spectral_features(df: pd.DataFrame) -> np.ndarray:
    # "direct spectral feature clustering" baseline: flatten logmel then PCA :contentReference[oaicite:6]{index=6}
    paths = df["logmel_path"].tolist()
    mats = []
    for p in paths:
        m = np.load(p).astype(np.float32)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        mats.append(m.reshape(-1))
    X = np.stack(mats, axis=0)
    # standardize and PCA to 50D
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=42)
    return pca.fit_transform(X)


def run_kmeans_grid(X: np.ndarray, k_min: int, k_max: int, seed: int) -> List[Tuple[str, np.ndarray]]:
    outs = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = km.fit_predict(X)
        outs.append((f"kmeans_k={k}", labels))
    return outs


def run_agglo_grid(X: np.ndarray, k_min: int, k_max: int) -> List[Tuple[str, np.ndarray]]:
    outs = []
    for k in range(k_min, k_max + 1):
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = ag.fit_predict(X)
        outs.append((f"agglo_k={k}_ward", labels))
    return outs


def run_dbscan_grid(X: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    outs = []
    for eps in [0.3, 0.5, 0.8, 1.0]:
        for ms in [5, 10]:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            outs.append((f"dbscan_eps={eps}_min={ms}", labels))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="data/processed_hard/dataset_clean_with_genre.csv")
    ap.add_argument("--data_dir", required=True, help="data/processed_hard")
    ap.add_argument("--beta_latent", required=True, help=".../results_beta_vae/latent_mu.npy")
    ap.add_argument("--ae_latent", required=True, help=".../results_audio_ae/latent.npy")
    ap.add_argument("--out_dir", required=True, help="data/processed_hard/results_hard_final")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "genre" not in df.columns:
        raise ValueError("CSV must contain 'genre' column for hard-task label metrics.")
    if "language" not in df.columns:
        raise ValueError("CSV must contain 'language' column.")
    if "logmel_path" not in df.columns:
        raise ValueError("CSV must contain 'logmel_path' column.")

    # True labels
    y_lang, lang_map = categorical_to_int(df["language"])
    y_genre, genre_map = categorical_to_int(df["genre"])
    y_joint, joint_map = categorical_to_int(df["language"].astype(str) + "__" + df["genre"].astype(str))

    # Features (representations)
    data_dir = Path(args.data_dir)

    X_audio = np.load(data_dir / "audio_features_clean.npy").astype(np.float32)          # (N,69)
    X_audio = (X_audio - X_audio.mean(axis=0, keepdims=True)) / (X_audio.std(axis=0, keepdims=True) + 1e-8)

    X_beta = np.load(args.beta_latent).astype(np.float32)                                # (N,latent)
    X_beta = (X_beta - X_beta.mean(axis=0, keepdims=True)) / (X_beta.std(axis=0, keepdims=True) + 1e-8)

    X_ae = np.load(args.ae_latent).astype(np.float32)                                    # (N,latent)
    X_ae = (X_ae - X_ae.mean(axis=0, keepdims=True)) / (X_ae.std(axis=0, keepdims=True) + 1e-8)

    # PCA baseline on audio_features (PCA + KMeans) 
    pca = PCA(n_components=min(16, X_audio.shape[1], X_audio.shape[0] - 1), random_state=args.seed)
    X_pca = pca.fit_transform(X_audio).astype(np.float32)
    X_pca = (X_pca - X_pca.mean(axis=0, keepdims=True)) / (X_pca.std(axis=0, keepdims=True) + 1e-8)

    # Direct spectral baseline (logmel flatten -> PCA) :contentReference[oaicite:8]{index=8}
    X_spec = load_direct_spectral_features(df).astype(np.float32)
    X_spec = (X_spec - X_spec.mean(axis=0, keepdims=True)) / (X_spec.std(axis=0, keepdims=True) + 1e-8)

    # Multimodal fusion: beta latent + onehot(language) + onehot(genre)
    one_lang = make_onehot(y_lang, len(lang_map))
    one_genre = make_onehot(y_genre, len(genre_map))
    X_multi = np.concatenate([X_beta, one_lang, one_genre], axis=1).astype(np.float32)
    X_multi = (X_multi - X_multi.mean(axis=0, keepdims=True)) / (X_multi.std(axis=0, keepdims=True) + 1e-8)

    reps = {
        "audio_features": X_audio,
        "pca_audio": X_pca,
        "autoencoder_latent": X_ae,
        "beta_vae_latent": X_beta,
        "direct_spectral": X_spec,
        "multimodal_beta+lang+genre": X_multi,
    }

    rows = []

    def evaluate_rep(rep_name: str, X: np.ndarray) -> None:
        # KMeans grid
        for param, labels in run_kmeans_grid(X, args.k_min, args.k_max, args.seed):
            cm = safe_cluster_metrics(X, labels)
            lm_lang = label_metrics(labels, y_lang)
            lm_genre = label_metrics(labels, y_genre)
            lm_joint = label_metrics(labels, y_joint)
            rows.append({
                "rep": rep_name,
                "algo": "kmeans",
                "param": param,
                **cm,
                "nmi_lang": lm_lang["nmi"],
                "ari_lang": lm_lang["ari"],
                "purity_lang": lm_lang["purity"],
                "nmi_genre": lm_genre["nmi"],
                "ari_genre": lm_genre["ari"],
                "purity_genre": lm_genre["purity"],
                "nmi_joint": lm_joint["nmi"],
                "ari_joint": lm_joint["ari"],
                "purity_joint": lm_joint["purity"],
            })

        # Agglo grid
        for param, labels in run_agglo_grid(X, args.k_min, args.k_max):
            cm = safe_cluster_metrics(X, labels)
            lm_lang = label_metrics(labels, y_lang)
            lm_genre = label_metrics(labels, y_genre)
            lm_joint = label_metrics(labels, y_joint)
            rows.append({
                "rep": rep_name,
                "algo": "agglomerative",
                "param": param,
                **cm,
                "nmi_lang": lm_lang["nmi"],
                "ari_lang": lm_lang["ari"],
                "purity_lang": lm_lang["purity"],
                "nmi_genre": lm_genre["nmi"],
                "ari_genre": lm_genre["ari"],
                "purity_genre": lm_genre["purity"],
                "nmi_joint": lm_joint["nmi"],
                "ari_joint": lm_joint["ari"],
                "purity_joint": lm_joint["purity"],
            })

        # DBSCAN grid
        for param, labels in run_dbscan_grid(X):
            cm = safe_cluster_metrics(X, labels)
            lm_lang = label_metrics(labels, y_lang)
            lm_genre = label_metrics(labels, y_genre)
            lm_joint = label_metrics(labels, y_joint)
            rows.append({
                "rep": rep_name,
                "algo": "dbscan",
                "param": param,
                **cm,
                "nmi_lang": lm_lang["nmi"],
                "ari_lang": lm_lang["ari"],
                "purity_lang": lm_lang["purity"],
                "nmi_genre": lm_genre["nmi"],
                "ari_genre": lm_genre["ari"],
                "purity_genre": lm_genre["purity"],
                "nmi_joint": lm_joint["nmi"],
                "ari_joint": lm_joint["ari"],
                "purity_joint": lm_joint["purity"],
            })

    # Evaluate all representations
    for name, X in reps.items():
        print("Evaluating:", name, "X shape:", X.shape)
        evaluate_rep(name, X)

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "metrics_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)

    # pick "best" run by silhouette (ties broken by nmi_joint)
    metrics_sorted = metrics.sort_values(
        by=["silhouette", "nmi_joint"],
        ascending=[False, False],
        na_position="last",
    )
    best = metrics_sorted.iloc[0].to_dict()
    (out_dir / "best_run.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Best run:", best["rep"], best["algo"], best["param"], "silhouette=", best["silhouette"])

    # Recompute labels for the best run (kmeans/agglo/dbscan)
    best_rep = best["rep"]
    Xbest = reps[best_rep]
    algo = best["algo"]
    param = str(best["param"])

    if algo == "kmeans":
        k = int(param.split("k=")[1])
        model = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
        labels = model.fit_predict(Xbest)
    elif algo == "agglomerative":
        k = int(param.split("k=")[1].split("_")[0])
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(Xbest)
    else:
        # dbscan_eps=0.5_min=10
        eps = float(param.split("eps=")[1].split("_")[0])
        ms = int(param.split("min=")[1])
        model = DBSCAN(eps=eps, min_samples=ms)
        labels = model.fit_predict(Xbest)

    # Save assignments
    out_assign = df.copy()
    out_assign["cluster"] = labels
    assign_path = out_dir / "cluster_assignments_best.csv"
    out_assign.to_csv(assign_path, index=False)
    print("Saved:", assign_path)

    # Crosstabs / distributions
    pd.crosstab(out_assign["cluster"], out_assign["language"]).to_csv(out_dir / "cluster_x_language.csv")
    pd.crosstab(out_assign["cluster"], out_assign["genre"]).to_csv(out_dir / "cluster_x_genre.csv")
    print("Saved crosstabs.")

    # 2D embedding plots (cluster, language, genre) - required visualizations :contentReference[oaicite:9]{index=9}
    X2 = embed_2d(Xbest, args.seed)
    np.save(out_dir / "embedding_2d.npy", X2.astype(np.float32))

    scatter_save(X2, labels, f"{best_rep} | best clusters", out_dir / "umap_or_tsne_by_cluster.png")
    scatter_save(X2, y_lang, f"{best_rep} | language", out_dir / "umap_or_tsne_by_language.png")
    scatter_save(X2, y_genre, f"{best_rep} | genre", out_dir / "umap_or_tsne_by_genre.png")
    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
