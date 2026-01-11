# scripts/make_hybrid_with_genre.py
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataset_clean_with_genre.csv")
    ap.add_argument("--audio_npy", default="data/processed_hard/audio_features_clean.npy")
    ap.add_argument("--lyrics_npy", default="data/processed_hard/results_hybrid/lyrics_features.npy")
    ap.add_argument("--out_dir", default="data/processed_hard/results_hybrid")
    ap.add_argument("--out_name", default="hybrid_features.npy")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    audio = np.load(args.audio_npy).astype(np.float32)
    lyrics = np.load(args.lyrics_npy).astype(np.float32)

    if len(df) != audio.shape[0] or len(df) != lyrics.shape[0]:
        raise ValueError(f"Row mismatch: df={len(df)} audio={audio.shape[0]} lyrics={lyrics.shape[0]}")

    genres = sorted(df["genre"].astype(str).unique().tolist())
    g2i = {g: i for i, g in enumerate(genres)}

    onehot = np.zeros((len(df), len(genres)), dtype=np.float32)
    for i, g in enumerate(df["genre"].astype(str)):
        onehot[i, g2i[g]] = 1.0

    hybrid = np.concatenate([audio, lyrics, onehot], axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / args.out_name, hybrid)

    (out_dir / "genre_vocab.json").write_text(json.dumps(genres, indent=2), encoding="utf-8")
    print(" Saved:", (out_dir / args.out_name).as_posix(), "shape:", hybrid.shape)
    print(" Saved:", (out_dir / "genre_vocab.json").as_posix(), "n_genres:", len(genres))

if __name__ == "__main__":
    main()
