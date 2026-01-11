import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main(
    in_csv="data/processed/dataset_clean.csv",
    out_dir="data/processed/splits",
    test_size=0.15,
    val_size=0.15,
    seed=42,
):
    df = pd.read_csv(in_csv)

    required = {"track_id", "segment_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {in_csv}: {missing}")

    track_ids = df["track_id"].astype(str).unique().tolist()
    if len(track_ids) < 10:
        raise RuntimeError(f"Not enough unique track_id to split: {len(track_ids)}")

    train_ids, test_ids = train_test_split(track_ids, test_size=test_size, random_state=seed)
    val_ratio = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_ratio, random_state=seed)

    os.makedirs(out_dir, exist_ok=True)

    def write_split(name, ids):
        split_df = df[df["track_id"].astype(str).isin(ids)].copy()
        if len(split_df) == 0:
            raise RuntimeError(f"{name} split is empty.")
        path = os.path.join(out_dir, f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"{name}: {len(split_df)} rows -> {path}")

    write_split("train", train_ids)
    write_split("val", val_ids)
    write_split("test", test_ids)

if __name__ == "__main__":
    main()
