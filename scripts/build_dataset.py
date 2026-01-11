import os
import json
import math
import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import yaml
import librosa
import soundfile as sf

try:
    from mutagen import File as mutagen_file
except Exception:
    mutagen_file = None


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif")


@dataclass
class SourceSpec:
    name: str
    root: str
    language: str


def iter_audio_files(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(AUDIO_EXTS):
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def get_duration_seconds(path: str) -> Optional[float]:
    # Fast path for wav/flac etc.
    try:
        info = sf.info(path)
        if info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    # Good for mp3/m4a if mutagen is available
    if mutagen_file is not None:
        try:
            mf = mutagen_file(path)
            if mf is not None and mf.info is not None and getattr(mf.info, "length", None):
                return float(mf.info.length)
        except Exception:
            pass

    # Fallback (may decode)
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        return None


def choose_segment_start(
    duration: float,
    seg_seconds: float,
    strategy: str,
    rng: random.Random
) -> float:
    if duration <= seg_seconds:
        return 0.0

    max_start = max(0.0, duration - seg_seconds)

    if strategy == "start":
        return 0.0
    if strategy == "center":
        return max(0.0, (duration - seg_seconds) / 2.0)
    if strategy == "random":
        return rng.uniform(0.0, max_start)

    raise ValueError(f"Unknown segment strategy: {strategy}")


def load_audio_segment(path: str, sr: int, offset: float, duration: float) -> Tuple[np.ndarray, int]:
    y, _sr = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
    return y, sr


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        return y[:target_len]
    pad = target_len - len(y)
    return np.pad(y, (0, pad), mode="constant")


def feature_columns() -> List[str]:
    cols = []
    cols += [f"mfcc_mean_{i}" for i in range(20)]
    cols += [f"mfcc_std_{i}" for i in range(20)]
    cols += [f"chroma_mean_{i}" for i in range(12)]
    cols += [f"chroma_std_{i}" for i in range(12)]
    cols += ["spec_centroid", "spec_rolloff", "zcr", "rms", "tempo"]
    return cols


def extract_feature_vector(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_std = chroma.std(axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Fix NumPy warning: ensure scalar
        tempo = float(np.asarray(tempo).item())
    except Exception:
        tempo = float("nan")

    feats = np.concatenate([
        mfcc_mean, mfcc_std,
        chroma_mean, chroma_std,
        np.array([centroid, rolloff, zcr, rms, tempo], dtype=np.float32)
    ]).astype(np.float32)

    return feats


def compute_logmel(y: np.ndarray, sr: int, n_mels: int = 128, hop_length: int = 512) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, power=2.0)
    logS = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return logS


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_sources_yaml(path: str) -> List[SourceSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    specs = []
    for s in data.get("sources", []):
        specs.append(SourceSpec(name=s["name"], root=s["root"], language=s["language"]))
    return specs


def balanced_select(
    per_source_files: Dict[str, List[str]],
    specs: List[SourceSpec],
    max_files_total: int,
    rng: random.Random
) -> List[Tuple[SourceSpec, str]]:
    """
    Try to select ~equal number from each source, then redistribute if some sources have fewer.
    """
    active_specs = [s for s in specs if s.name in per_source_files and len(per_source_files[s.name]) > 0]
    if not active_specs:
        return []

    n = len(active_specs)
    base = max_files_total // n
    rem = max_files_total % n

    # Initial quotas
    quotas = {s.name: base for s in active_specs}
    # Distribute remainder
    for i in range(rem):
        quotas[active_specs[i].name] += 1

    selected: List[Tuple[SourceSpec, str]] = []
    remaining_pool: List[Tuple[SourceSpec, str]] = []

    # First pass: take up to quota
    for s in active_specs:
        files = per_source_files[s.name]
        take = min(len(files), quotas[s.name])
        for p in files[:take]:
            selected.append((s, p))

        # leftover goes to pool for redistribution
        for p in files[take:]:
            remaining_pool.append((s, p))

    # If still not enough (due to small sources), top-up from remaining pool
    if len(selected) < max_files_total and remaining_pool:
        rng.shuffle(remaining_pool)
        need = max_files_total - len(selected)
        selected.extend(remaining_pool[:need])

    return selected[:max_files_total]


def build_dataset(
    sources_yaml: str = "sources.yaml",
    out_dir: str = "data/processed",
    out_csv: str = "dataset.csv",
    out_audio_npy: str = "audio_features.npy",
    out_cols_json: str = "audio_feature_cols.json",
    save_logmels: bool = False,
    mels_subdir: str = "mels",
    sr: int = 22050,
    seg_seconds: float = 10.0,
    num_segments_per_track: int = 1,
    seg_strategy: str = "random",  # random|center|start
    min_duration: float = 3.0,
    seed: int = 42,
    max_files_total: int = 1200,
    balance_sources: bool = True,
) -> None:
    rng = random.Random(seed)

    specs = read_sources_yaml(sources_yaml)
    if not specs:
        raise RuntimeError(f"No sources found in {sources_yaml}")

    out_dir_abs = os.path.abspath(out_dir)
    ensure_dir(out_dir_abs)
    if save_logmels:
        ensure_dir(os.path.join(out_dir_abs, mels_subdir))

    # Save feature column names
    cols = feature_columns()
    with open(os.path.join(out_dir_abs, out_cols_json), "w", encoding="utf-8") as f:
        json.dump(cols, f, indent=2)

    # Scan files per source
    per_source_files: Dict[str, List[str]] = {}
    active_specs: List[SourceSpec] = []

    for spec in specs:
        if not os.path.exists(spec.root):
            print(f"[WARN] Missing source root, skipping: {spec.root}")
            continue
        files = iter_audio_files(spec.root)
        if files:
            # Shuffle deterministically per source
            local_rng = random.Random(f"{seed}:{spec.name}")
            local_rng.shuffle(files)
            per_source_files[spec.name] = files
            active_specs.append(spec)

    if not active_specs:
        raise RuntimeError("No audio files found. Check sources.yaml roots and that audio exists.")

    total_found = sum(len(per_source_files[s.name]) for s in active_specs)
    print(f"Found {total_found} audio files across sources.")

    # Select subset (DEFAULT: 1200 files total)
    if max_files_total is not None and max_files_total > 0:
        max_files_total = min(max_files_total, total_found)
        if balance_sources and len(active_specs) > 1:
            all_items = balanced_select(per_source_files, active_specs, max_files_total, rng)
        else:
            pooled: List[Tuple[SourceSpec, str]] = []
            for s in active_specs:
                for p in per_source_files[s.name]:
                    pooled.append((s, p))
            rng.shuffle(pooled)
            all_items = pooled[:max_files_total]
        print(f"Using {len(all_items)} files (max_files_total={max_files_total}, balance_sources={balance_sources}).")
    else:
        # Use everything
        all_items = []
        for s in active_specs:
            for p in per_source_files[s.name]:
                all_items.append((s, p))
        print(f"Using ALL files: {len(all_items)}")

    rows: List[Dict] = []
    feats: List[np.ndarray] = []
    target_len = int(sr * seg_seconds)

    for spec, path in tqdm(all_items, desc="Building dataset"):
        dur = get_duration_seconds(path)
        if dur is None:
            dur = float("nan")

        if not math.isnan(dur) and dur < min_duration:
            continue

        rel = os.path.relpath(path, spec.root)
        base = f"{spec.name}::{rel}"
        track_base_id = stable_id(base)

        for seg_idx in range(num_segments_per_track):
            # deterministic per track + segment
            local_rng = random.Random(f"{seed}:{track_base_id}:{seg_idx}")

            if math.isnan(dur):
                offset = 0.0
            else:
                offset = choose_segment_start(dur, seg_seconds, seg_strategy, local_rng)

            try:
                y, _ = load_audio_segment(path, sr=sr, offset=offset, duration=seg_seconds)
            except Exception:
                continue

            y = pad_or_trim(y, target_len)

            # normalize
            maxv = float(np.max(np.abs(y))) if len(y) else 0.0
            if maxv > 0:
                y = (y / maxv).astype(np.float32)
            else:
                y = y.astype(np.float32)

            fv = extract_feature_vector(y, sr)
            feats.append(fv)

            seg_id = f"{track_base_id}_s{seg_idx:02d}"
            row = {
                "track_id": track_base_id,
                "segment_id": seg_id,
                "segment_index": seg_idx,
                "source": spec.name,
                "language": spec.language,
                "audio_path": os.path.abspath(path),
                "sr": sr,
                "segment_seconds": float(seg_seconds),
                "segment_strategy": seg_strategy,
                "segment_offset_seconds": float(offset),
                "duration_seconds": None if math.isnan(dur) else float(dur),
            }

            if save_logmels:
                logmel = compute_logmel(y, sr=sr, n_mels=128, hop_length=512)
                mel_path = os.path.join(out_dir_abs, mels_subdir, f"{seg_id}.npy")
                np.save(mel_path, logmel)
                row["logmel_path"] = mel_path
                row["logmel_n_mels"] = 128
                row["logmel_hop_length"] = 512

            rows.append(row)

    if not rows:
        raise RuntimeError("No usable items were processed (all skipped/failed).")

    df = pd.DataFrame(rows)
    X = np.vstack(feats).astype(np.float32)

    df_path = os.path.join(out_dir_abs, out_csv)
    npy_path = os.path.join(out_dir_abs, out_audio_npy)

    df.to_csv(df_path, index=False)
    np.save(npy_path, X)

    print("\nSaved:")
    print(f"- CSV manifest: {df_path}  (rows={len(df)})")
    print(f"- Audio features: {npy_path}  (shape={X.shape})")
    print(f"- Feature columns: {os.path.join(out_dir_abs, out_cols_json)}")
    if save_logmels:
        print(f"- Log-mels folder: {os.path.join(out_dir_abs, mels_subdir)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="sources.yaml")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--save_logmels", action="store_true")

    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--seg_seconds", type=float, default=10.0)
    ap.add_argument("--num_segments", "--num_segments_per_track", dest="num_segments", type=int, default=1)
    ap.add_argument("--seg_strategy", type=str, default="random", choices=["random", "center", "start"])
    ap.add_argument("--min_duration", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)

    # NEW: limit how many files to use
    ap.add_argument("--max_files_total", type=int, default=1200)
    ap.add_argument("--no_balance_sources", action="store_true", help="Disable balancing across sources")

    args = ap.parse_args()

    build_dataset(
        sources_yaml=args.sources,
        out_dir=args.out_dir,
        save_logmels=args.save_logmels,
        sr=args.sr,
        seg_seconds=args.seg_seconds,
        num_segments_per_track=args.num_segments,
        seg_strategy=args.seg_strategy,
        min_duration=args.min_duration,
        seed=args.seed,
        max_files_total=args.max_files_total,
        balance_sources=(not args.no_balance_sources),
    )

