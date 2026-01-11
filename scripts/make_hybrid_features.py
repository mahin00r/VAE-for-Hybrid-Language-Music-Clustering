import os
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--audio_npy", default="audio_features_clean.npy")
    ap.add_argument("--lyrics_npy", default="results_hybrid/lyrics_features.npy")
    ap.add_argument("--out", default="results_hybrid/hybrid_features.npy")
    args = ap.parse_args()

    ds = args.dataset_dir
    A = np.load(os.path.join(ds, args.audio_npy)).astype(np.float32)
    L = np.load(os.path.join(ds, args.lyrics_npy)).astype(np.float32)

    n = min(A.shape[0], L.shape[0])
    A, L = A[:n], L[:n]

    # scale each block then concat (important)
    A = StandardScaler().fit_transform(A).astype(np.float32)
    L = StandardScaler().fit_transform(L).astype(np.float32)

    H = np.concatenate([A, L], axis=1).astype(np.float32)

    out_path = os.path.join(ds, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, H)

    print("Saved hybrid:", out_path, "shape:", H.shape)

if __name__ == "__main__":
    main()
