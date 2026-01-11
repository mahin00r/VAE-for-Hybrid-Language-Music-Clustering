# scripts/make_text_for_each_track.py
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dataset_clean_with_genre.csv")
    ap.add_argument("--out_dir", required=True, help="Folder to write .txt files")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"audio_path", "language", "genre"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    n = 0
    for _, r in df.iterrows():
        stem = Path(str(r["audio_path"])).stem.lower()  # must match build_lyrics_embeddings.py key logic
        lang = str(r["language"]).strip()
        genre = str(r["genre"]).strip()

        # Text content: simple but non-empty, consistent
        text = f"language {lang} genre {genre} {genre} music"
        (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        n += 1

    print(f"✅ Wrote {n} text files into: {out_dir.resolve()}")
    print("Example:", next(out_dir.glob("*.txt")).name if any(out_dir.glob("*.txt")) else "none")

if __name__ == "__main__":
    main()
