import os, json, argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    args = ap.parse_args()

    in_dir = args.in_dir

    df_path = os.path.join(in_dir, "dataset.csv")
    X_path = os.path.join(in_dir, "audio_features.npy")
    cols_path = os.path.join(in_dir, "audio_feature_cols.json")

    df = pd.read_csv(df_path)
    X = np.load(X_path)
    cols = json.load(open(cols_path, "r", encoding="utf-8"))

    n = min(len(df), X.shape[0])
    df = df.iloc[:n].reset_index(drop=True)
    X = X[:n]

    # Drop duplicates
    keep_idx = df.drop_duplicates(subset=["segment_id"], keep="first").index.to_numpy()
    df = df.iloc[keep_idx].reset_index(drop=True)
    X = X[keep_idx]

    # Drop NaN/inf features
    good = np.isfinite(X).all(axis=1)
    df = df.loc[good].reset_index(drop=True)
    X = X[good]

    df.to_csv(os.path.join(in_dir, "dataset_clean.csv"), index=False)
    np.save(os.path.join(in_dir, "audio_features_clean.npy"), X)

    with open(os.path.join(in_dir, "audio_feature_cols_clean.json"), "w", encoding="utf-8") as f:
        json.dump(cols, f, indent=2)

    print("✅ Clean saved:")
    print(os.path.join(in_dir, "dataset_clean.csv"))
    print(os.path.join(in_dir, "audio_features_clean.npy"))
    print("Rows:", len(df), "X shape:", X.shape)

if __name__ == "__main__":
    main()
