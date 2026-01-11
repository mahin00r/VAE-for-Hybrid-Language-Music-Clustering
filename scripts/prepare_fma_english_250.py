import os
import random
import shutil
import argparse
import pandas as pd

AUDIO_EXT = ".mp3"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_fma_mp3(fma_small_root: str, track_id: int) -> str | None:
    tid = f"{track_id:06d}"
    sub = tid[:3]
    p = os.path.join(fma_small_root, sub, tid + AUDIO_EXT)
    return p if os.path.exists(p) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fma_small_root", default="data/raw/fma_downloads/fma_small/fma_small")
    ap.add_argument("--tracks_csv", default="data/raw/fma_downloads/fma_metadata/fma_metadata/tracks.csv")
    ap.add_argument("--out_root", default="data/raw/english_fma")
    ap.add_argument("--total", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.tracks_csv):
        raise FileNotFoundError(f"Missing tracks.csv: {args.tracks_csv}")
    if not os.path.isdir(args.fma_small_root):
        raise FileNotFoundError(f"Missing fma_small root: {args.fma_small_root}")

    df = pd.read_csv(args.tracks_csv, header=[0,1], index_col=0)

    # Filter to subset == 'small' (important)
    subset_col = ("set", "subset")
    genre_col = ("track", "genre_top")
    if subset_col not in df.columns or genre_col not in df.columns:
        raise RuntimeError(f"Expected columns {subset_col} and {genre_col} not found in tracks.csv")

    df = df[df[subset_col].astype(str) == "small"]
    df = df[df[genre_col].notna()]
    df[genre_col] = df[genre_col].astype(str)

    genres = sorted(df[genre_col].unique().tolist())
    if len(genres) == 0:
        raise RuntimeError("No genres found after filtering tracks.csv")

    base = args.total // len(genres)
    extra = args.total % len(genres)

    print("Genres:", genres)
    print(f"Target total={args.total}, base_per_genre={base}, remainder={extra}")

    ensure_dir(args.out_root)

    chosen_all = []
    for gi, g in enumerate(genres):
        want = base + (1 if gi < extra else 0)
        ids = df[df[genre_col] == g].index.astype(int).tolist()
        random.shuffle(ids)

        picked = []
        for tid in ids:
            src = find_fma_mp3(args.fma_small_root, tid)
            if src is None:
                continue
            picked.append((g, tid, src))
            if len(picked) >= want:
                break

        if len(picked) < want:
            print(f"[WARN] Genre {g}: wanted {want} but found {len(picked)}")

        chosen_all.extend(picked)

    # If short (rare), top up from any genre
    if len(chosen_all) < args.total:
        need = args.total - len(chosen_all)
        print(f"[WARN] Short by {need}; topping up from any genre...")
        all_ids = df.index.astype(int).tolist()
        random.shuffle(all_ids)
        existing = set((tid for _, tid, _ in chosen_all))
        for tid in all_ids:
            if tid in existing:
                continue
            g = df.loc[tid, genre_col]
            src = find_fma_mp3(args.fma_small_root, tid)
            if src is None:
                continue
            chosen_all.append((str(g), int(tid), src))
            if len(chosen_all) >= args.total:
                break

    # Copy files
    copied = 0
    for i, (g, tid, src) in enumerate(chosen_all, start=1):
        out_dir = os.path.join(args.out_root, g)
        ensure_dir(out_dir)
        dst = os.path.join(out_dir, f"{g}_{tid:06d}.mp3")
        shutil.copy2(src, dst)
        copied += 1

    print("Done. Copied:", copied)
    print("Output:", args.out_root)

if __name__ == "__main__":
    main()
