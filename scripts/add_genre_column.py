import os
import argparse
import pandas as pd

AUDIO_KEYS = ("bangla", "english")

def infer_genre_from_path(p: str) -> str:
    # Normalize separators
    parts = p.replace("\\", "/").split("/")
    # Example expected:
    # .../raw_balanced_500_hard/bangla/Pop/bn_0001.wav
    # .../raw_balanced_500_hard/english/Rock/Rock_000123.mp3
    for key in AUDIO_KEYS:
        if key in parts:
            i = parts.index(key)
            # genre is the folder right after bangla/english
            if i + 1 < len(parts) - 1:
                return parts[i + 1]
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/processed_hard/dataset_clean.csv")
    ap.add_argument("--out_csv", default="data/processed_hard/dataset_clean_with_genre.csv")
    ap.add_argument("--path_col", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # auto-detect the path column
    if args.path_col is None:
        candidates = [c for c in df.columns if "path" in c.lower()]
        if not candidates:
            raise RuntimeError("No column containing 'path' found. Use --path_col explicitly.")
        path_col = candidates[0]
    else:
        path_col = args.path_col

    df["genre"] = df[path_col].astype(str).apply(infer_genre_from_path)

    # sanity check
    if (df["genre"] == "unknown").mean() > 0.2:
        print("[WARN] Many genres are 'unknown'. Your folder layout may not match expected pattern.")
        print("Example paths:")
        print(df[path_col].head(10).to_string(index=False))

    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print("Top genre counts:")
    print(df["genre"].value_counts().head(20))
    print("Unknown count:", (df["genre"] == "unknown").sum())

if __name__ == "__main__":
    main()
