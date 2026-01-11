import os
import json
import numpy as np
import pandas as pd

DATASET_CSV = "data/processed/dataset.csv"
FEATURES_NPY = "data/processed/audio_features.npy"
COLS_JSON = "data/processed/audio_feature_cols.json"

OUT_CSV = "data/processed/dataset_clean.csv"
OUT_NPY = "data/processed/audio_features_clean.npy"
OUT_COLS_JSON = "data/processed/audio_feature_cols_clean.json"

def main():
    if not os.path.exists(DATASET_CSV):
        raise FileNotFoundError(f"Missing: {DATASET_CSV}")
    if not os.path.exists(FEATURES_NPY):
        raise FileNotFoundError(f"Missing: {FEATURES_NPY}")
    if not os.path.exists(COLS_JSON):
        raise FileNotFoundError(f"Missing: {COLS_JSON}")

    df = pd.read_csv(DATASET_CSV)
    X = np.load(FEATURES_NPY)
    cols = json.load(open(COLS_JSON, "r", encoding="utf-8"))

    # Safety: align lengths if anything is off
    n = min(len(df), X.shape[0])
    df = df.iloc[:n].reset_index(drop=True)
    X = X[:n]

    before = len(df)

    # 1) Drop duplicate segments (keep first)
    # (If you don't have duplicates, this changes nothing.)
    keep_idx = df.drop_duplicates(subset=["segment_id"], keep="first").index.to_numpy()
    df = df.iloc[keep_idx].reset_index(drop=True)
    X = X[keep_idx]

    # 2) Remove rows with NaN/inf in features
    good = np.isfinite(X).all(axis=1)
    df = df.loc[good].reset_index(drop=True)
    X = X[good]

    after = len(df)

    # Save cleaned outputs
    df.to_csv(OUT_CSV, index=False)
    np.save(OUT_NPY, X)

    # Save columns list too (same as before, just mirrored for convenience)
    with open(OUT_COLS_JSON, "w", encoding="utf-8") as f:
        json.dump(cols, f, indent=2)

    print("Cleaning complete")
    print(f"Before rows: {before}")
    print(f"After rows : {after}")
    print(f"Saved CSV  : {OUT_CSV}")
    print(f"Saved NPY  : {OUT_NPY}")

if __name__ == "__main__":
    main()

