import os
import random
import shutil
import argparse
import yaml

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif")

def iter_audio_files(root: str):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_copy(src: str, dst_dir: str, new_name: str):
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, new_name)
    shutil.copy2(src, dst)
    return dst

def guess_genre_from_parent(src_path: str, dataset_root: str) -> str:
    """
    Genre = the folder that contains the file (its parent dir name).
    This works for:
      - bangla/.../<Genre>/<file>
      - english_fma/<Genre>/<file>
    """
    parent = os.path.basename(os.path.dirname(src_path))
    # If somehow parent is the root folder itself, label unknown
    if os.path.abspath(os.path.dirname(src_path)) == os.path.abspath(dataset_root):
        return "unknown"
    # Clean up any bad chars for folder names
    parent = parent.strip().replace("/", "_").replace("\\", "_")
    return parent if parent else "unknown"

def write_sources_yaml(out_yaml: str, bangla_root: str, english_root: str):
    data = {
        "sources": [
            {"name": "bangla", "root": bangla_root.replace("\\", "/"), "language": "bn"},
            {"name": "english", "root": english_root.replace("\\", "/"), "language": "en"},
        ]
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bangla_root", default="data/raw/bangla")
    ap.add_argument("--english_root", default="data/raw/english_fma")
    ap.add_argument("--out_root", default="data/raw_balanced_500_hard")
    ap.add_argument("--n_bangla", type=int, default=250)
    ap.add_argument("--n_english", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_sources_yaml", default="sources_500_hard.yaml")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    bangla_files = iter_audio_files(args.bangla_root)
    english_files = iter_audio_files(args.english_root)

    print("Found bangla files:", len(bangla_files))
    print("Found english files:", len(english_files))

    if len(bangla_files) < args.n_bangla:
        raise RuntimeError(f"Not enough bangla files: have {len(bangla_files)} need {args.n_bangla}")
    if len(english_files) < args.n_english:
        raise RuntimeError(f"Not enough english files: have {len(english_files)} need {args.n_english}")

    rng.shuffle(bangla_files)
    rng.shuffle(english_files)

    bangla_pick = bangla_files[: args.n_bangla]
    english_pick = english_files[: args.n_english]

    out_bangla = os.path.join(args.out_root, "bangla")
    out_english = os.path.join(args.out_root, "english")

    # clean output folder if already exists
    if os.path.exists(args.out_root):
        shutil.rmtree(args.out_root)
    ensure_dir(out_bangla)
    ensure_dir(out_english)

    # Copy Bangla preserving genre folder = parent directory name
    for i, src in enumerate(bangla_pick, start=1):
        genre = guess_genre_from_parent(src, args.bangla_root)
        ext = os.path.splitext(src)[1].lower()
        new_name = f"bn_{i:04d}{ext}"
        dst_dir = os.path.join(out_bangla, genre)
        safe_copy(src, dst_dir, new_name)

    # Copy English preserving genre folder = parent directory name
    for i, src in enumerate(english_pick, start=1):
        genre = guess_genre_from_parent(src, args.english_root)
        ext = os.path.splitext(src)[1].lower()
        new_name = f"en_{i:04d}{ext}"
        dst_dir = os.path.join(out_english, genre)
        safe_copy(src, dst_dir, new_name)

    # Write sources yaml pointing at the *top* folders
    write_sources_yaml(args.out_sources_yaml, out_bangla, out_english)

    print("\n✅ Balanced subset created (with genre folders):")
    print("Bangla root:", out_bangla)
    print("English root:", out_english)
    print("sources yaml ->", args.out_sources_yaml)

if __name__ == "__main__":
    main()
