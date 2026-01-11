import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NpyFeatureDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.from_numpy(self.X[idx]), idx


class AudioAE(nn.Module):
    def __init__(self, in_dim: int = 69, latent_dim: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, in_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat, z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="e.g. data/processed_hard (contains audio_features_clean.npy)")
    ap.add_argument("--splits_dir", required=True, help="e.g. data/processed_hard/splits")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "audio_features_clean.npy")  # (N,69)
    assert X.ndim == 2, X.shape

    # standardize
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    Xs = (X - mean) / std
    np.save(out_dir / "standardizer.npy", np.stack([mean.squeeze(), std.squeeze()], axis=0).astype(np.float32))

    # splits by reading split CSV lengths (we assume splits were created from same ordering of dataset_clean.csv,
    # but easiest is: train/val indices are first Ntrain rows etc NOT safe.
    # Instead: we train on full X (unsupervised baseline) and still produce latent for all.
    ds = NpyFeatureDataset(Xs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioAE(in_dim=Xs.shape[1], latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "loss"])
        w.writeheader()

    best = float("inf")
    bad = 0
    best_path = out_dir / "audio_ae.pt"

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for xb, _ in dl:
            xb = xb.to(device)
            xhat, _ = model(xb)
            loss = F.mse_loss(xhat, xb, reduction="mean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        loss_ep = tot / max(n, 1)
        print(f"epoch {ep:03d} loss={loss_ep:.6f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "loss"])
            w.writerow({"epoch": ep, "loss": loss_ep})

        if loss_ep < best - 1e-6:
            best = loss_ep
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # export latent for all rows
    dl_all = DataLoader(NpyFeatureDataset(Xs), batch_size=256, shuffle=False, num_workers=0)
    zs = []
    with torch.no_grad():
        for xb, _ in dl_all:
            xb = xb.to(device)
            _, z = model(xb)
            zs.append(z.cpu().numpy())
    Z = np.concatenate(zs, axis=0)
    np.save(out_dir / "latent.npy", Z)
    print("Saved:", out_dir / "latent.npy", "shape:", Z.shape)


if __name__ == "__main__":
    main()
