import os, argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    track_ids = df["track_id"].astype(str).unique().tolist()

    train_ids, test_ids = train_test_split(track_ids, test_size=args.test_size, random_state=args.seed)
    val_ratio = args.val_size / (1.0 - args.test_size)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_ratio, random_state=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, ids):
        s = df[df["track_id"].astype(str).isin(ids)].copy()
        path = os.path.join(args.out_dir, f"{name}.csv")
        s.to_csv(path, index=False)
        print(name, "rows:", len(s), "->", path)

    write("train", train_ids)
    write("val", val_ids)
    write("test", test_ids)

if __name__ == "__main__":
    main()
