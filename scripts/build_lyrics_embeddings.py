import argparse
import os
from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


LYR_EXTS = (".txt", ".lyrics", ".lyric")


def _norm_key(s: str) -> str:
    s = s.strip().lower()
    # keep letters/numbers/_/-
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s


def index_lyrics_files(lyrics_root: Path) -> dict:
    """
    Build {basename_without_ext_normalized: full_path} index.
    """
    idx = {}
    if not lyrics_root.exists():
        return idx

    for p in lyrics_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in LYR_EXTS:
            key = _norm_key(p.stem)
            if key and key not in idx:
                idx[key] = p
    return idx


def load_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="e.g., data/processed_hard")
    ap.add_argument("--lyrics_root", required=True, help="Folder containing lyrics .txt files")
    ap.add_argument("--dim", type=int, default=64, help="Lyrics embedding dimension (TF-IDF features)")
    ap.add_argument("--max_df", type=float, default=0.95)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--ngram_max", type=int, default=1)
    ap.add_argument("--out_subdir", default="results_hybrid", help="Output subfolder under dataset_dir")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    lyrics_root = Path(args.lyrics_root)
    out_dir = dataset_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # prefer dataset_clean_with_genre.csv, else dataset_clean.csv, else dataset.csv
    csv_path = None
    for name in ["dataset_clean_with_genre.csv", "dataset_clean.csv", "dataset.csv"]:
        p = dataset_dir / name
        if p.exists():
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(f"No dataset CSV found inside {dataset_dir}")

    df = pd.read_csv(csv_path)

    # Need a path column to match audio -> lyrics
    if "audio_path" not in df.columns:
        raise ValueError(f"{csv_path} must include 'audio_path' column.")

    # Build lyrics file index
    lyric_index = index_lyrics_files(lyrics_root)

    # Collect texts (match by audio basename)
    texts = []
    missing = 0

    # for debug: show a few sample basenames
    sample_keys = []
    for i, apath in enumerate(df["audio_path"].astype(str).tolist()):
        stem = Path(apath).stem
        key = _norm_key(stem)
        if i < 10:
            sample_keys.append(key)

        lp = lyric_index.get(key)
        if lp is None:
            texts.append("")
            missing += 1
        else:
            texts.append(load_text(lp))

    print(f"Lyrics missing for {missing}/{len(texts)} rows")
    print("Sample audio keys:", sample_keys[:10])

    # If EVERYTHING is missing or empty, write zeros and exit gracefully
    nonempty = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(nonempty) == 0:
        X = np.zeros((len(texts), args.dim), dtype=np.float32)
        np.save(out_dir / "lyrics_features.npy", X)
        with open(out_dir / "lyrics_feature_cols.json", "w", encoding="utf-8") as f:
            json.dump([f"tfidf_{i}" for i in range(args.dim)], f, indent=2)
        np.save(out_dir / "lyrics_missing_mask.npy", np.ones((len(texts),), dtype=np.uint8))
        print(f"[WARN] No lyrics found at all. Saved zero lyrics features: {out_dir / 'lyrics_features.npy'} shape={X.shape}")
        return

    # Vectorize with TF-IDF
    vec = TfidfVectorizer(
        max_features=args.dim,
        max_df=args.max_df,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
        stop_words=None,  # keep None for multilingual safety
    )

    # Fit-transform on all texts (empty texts are fine; vocab comes from non-empty)
    X = vec.fit_transform(texts).toarray().astype(np.float32)

    # Ensure fixed dim by padding (if vocab smaller than dim)
    if X.shape[1] < args.dim:
        pad = np.zeros((X.shape[0], args.dim - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=1)

    cols = [f"tfidf_{i}" for i in range(X.shape[1])]
    np.save(out_dir / "lyrics_features.npy", X)
    with open(out_dir / "lyrics_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(cols, f, indent=2)

    missing_mask = np.array([1 if (not t.strip()) else 0 for t in texts], dtype=np.uint8)
    np.save(out_dir / "lyrics_missing_mask.npy", missing_mask)

    print(f"Saved: {out_dir / 'lyrics_features.npy'} shape={X.shape}")
    print(f"Saved: {out_dir / 'lyrics_missing_mask.npy'} (1=missing) missing={missing_mask.sum()}")


if __name__ == "__main__":
    main()
