import os
import random
import shutil
import argparse
import pandas as pd

AUDIO_EXT = ".mp3"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_fma_mp3(fma_small_root: str, track_id: int) -> str | None:
    # FMA structure: fma_small/000/000002.mp3 etc (6-digit)
    tid = f"{track_id:06d}"
    sub = tid[:3]
    p = os.path.join(fma_small_root, sub, tid + AUDIO_EXT)
    return p if os.path.exists(p) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fma_small_root", default="data/raw/fma_downloads/fma_small")
    ap.add_argument("--tracks_csv", default="data/raw/fma_downloads/fma_metadata/tracks.csv")
    ap.add_argument("--out_root", default="data/raw/english_fma")
    ap.add_argument("--per_genre", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.tracks_csv):
        raise FileNotFoundError(f"Missing tracks.csv at: {args.tracks_csv}")
    if not os.path.isdir(args.fma_small_root):
        raise FileNotFoundError(f"Missing fma_small folder at: {args.fma_small_root}")

    # tracks.csv has multi-index columns; pandas can load it and we pick genre_top
    df = pd.read_csv(args.tracks_csv, header=[0, 1], index_col=0)
    # genre_top is in column ("track", "genre_top")
    if ("track", "genre_top") not in df.columns:
        raise RuntimeError("Could not find ('track','genre_top') in tracks.csv")

    genre_top = df[("track", "genre_top")].dropna()
    genre_top = genre_top.astype(str)

    # FMA small has 8 top genres typically; we will use whatever exists
    genres = sorted(genre_top.unique().tolist())

    print("Found genres:", genres)

    copied_total = 0
    for g in genres:
        ids = genre_top[genre_top == g].index.astype(int).tolist()
        random.shuffle(ids)

        out_dir = os.path.join(args.out_root, g)
        ensure_dir(out_dir)

        chosen = []
        for tid in ids:
            src = find_fma_mp3(args.fma_small_root, tid)
            if src is None:
                continue
            chosen.append((tid, src))
            if len(chosen) >= args.per_genre:
                break

        if len(chosen) == 0:
            print(f"[WARN] No files found for genre={g}")
            continue

        for i, (tid, src) in enumerate(chosen, start=1):
            dst = os.path.join(out_dir, f"{g}_{tid:06d}.mp3")
            shutil.copy2(src, dst)

        print(f"Genre {g}: copied {len(chosen)} files -> {out_dir}")
        copied_total += len(chosen)

    print("Done. Total copied:", copied_total)
    print("Output root:", args.out_root)

if __name__ == "__main__":
    main()
