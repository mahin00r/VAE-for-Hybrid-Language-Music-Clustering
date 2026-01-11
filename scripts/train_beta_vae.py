import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Standardizer:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + 1e-8) + self.mean


def compute_global_standardizer(paths: List[str], max_items: Optional[int] = None) -> Standardizer:
    """
    Compute global mean/std across ALL values of all spectrograms in `paths`.
    For ~500 items, full pass is fine.
    """
    vals_sum = 0.0
    vals_sumsq = 0.0
    count = 0

    use_paths = paths if max_items is None else paths[:max_items]
    for p in use_paths:
        arr = np.load(p)  # (F, T) or (1, F, T)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        vals_sum += float(arr.sum())
        vals_sumsq += float((arr * arr).sum())
        count += arr.size

    mean = vals_sum / max(count, 1)
    var = vals_sumsq / max(count, 1) - mean * mean
    std = float(np.sqrt(max(var, 1e-8)))
    return Standardizer(mean=mean, std=std)


# -----------------------------
# Dataset
# -----------------------------
class LogMelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, standardizer: Standardizer):
        self.df = df.reset_index(drop=True)
        self.standardizer = standardizer
        if "logmel_path" not in self.df.columns:
            raise ValueError("CSV must contain a 'logmel_path' column.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        p = self.df.loc[idx, "logmel_path"]
        x = np.load(p).astype(np.float32)  # (F, T) or (1,F,T)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        x = self.standardizer.transform(x)
        x = torch.from_numpy(x).unsqueeze(0)  # (1, F, T)
        return x, idx


# -----------------------------
# Beta-VAE model (Conv)
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim: int, in_ch: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=4, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),     # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),    # /2
            nn.ReLU(inplace=True),
        )

        # infer flatten size by dummy forward on expected size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 130)
            h = self.conv(dummy)
            self._h_shape = h.shape  # (1, C, H, W)
            flat = int(np.prod(h.shape[1:]))

        self.fc = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.flatten(1)
        h = self.fc(h)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, h_shape: torch.Size):
        super().__init__()
        _, C, H, W = h_shape
        self.C, self.H, self.W = C, H, W
        flat = C * H * W

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, flat),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(C, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # no activation: we reconstruct standardized real values
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), self.C, self.H, self.W)
        xhat = self.deconv(h)
        return xhat


class BetaVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.enc = Encoder(latent_dim=latent_dim)
        self.dec = Decoder(latent_dim=latent_dim, h_shape=self.enc._h_shape)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)

        # FIX: force recon size to match input exactly (handles 128 vs 130)
        if xhat.shape[-2:] != x.shape[-2:]:
            xhat = F.interpolate(xhat, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return xhat, mu, logvar


def vae_loss(x: torch.Tensor, xhat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float):
    """
    Reconstruction + beta*KL.
    This function also guarantees xhat matches x spatial shape (safety net).
    """
    if xhat.shape[-2:] != x.shape[-2:]:
        xhat = F.interpolate(xhat, size=x.shape[-2:], mode="bilinear", align_corners=False)

    recon = F.mse_loss(xhat, x, reduction="mean")
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    return loss, recon, kl


def save_recon_examples(
    model: BetaVAE,
    ds: LogMelDataset,
    device: torch.device,
    out_path: Path,
    n: int = 8,
) -> None:
    model.eval()
    n = min(n, len(ds))
    xs = []
    xhats = []
    with torch.no_grad():
        for i in range(n):
            x, _ = ds[i]
            x = x.unsqueeze(0).to(device)
            xhat, _, _ = model(x)
            xs.append(x.cpu().numpy()[0, 0])
            xhats.append(xhat.cpu().numpy()[0, 0])

    fig, axes = plt.subplots(n, 2, figsize=(8, 2 * n))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        axes[i, 0].imshow(xs[i], aspect="auto", origin="lower")
        axes[i, 0].set_title("Original (std)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(xhats[i], aspect="auto", origin="lower")
        axes[i, 1].set_title("Reconstruction (std)")
        axes[i, 1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def read_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def merge_splits_with_master(master: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    # If split already has full columns, keep it.
    if "logmel_path" in split_df.columns:
        return split_df

    if "segment_id" in split_df.columns and "segment_id" in master.columns:
        out = split_df.merge(master, on="segment_id", how="left")
        return out

    raise ValueError("Split CSV must contain either 'logmel_path' or 'segment_id' to merge.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dataset_clean_with_genre.csv (must include logmel_path).")
    ap.add_argument("--splits_dir", required=True, help="Directory containing train.csv and val.csv.")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.csv)
    train_df = read_split_csv(Path(args.splits_dir) / "train.csv")
    val_df = read_split_csv(Path(args.splits_dir) / "val.csv")

    train_df = merge_splits_with_master(master, train_df)
    val_df = merge_splits_with_master(master, val_df)

    # standardizer from train only
    stdz = compute_global_standardizer(train_df["logmel_path"].tolist())
    np.save(out_dir / "standardizer.npy", np.array([stdz.mean, stdz.std], dtype=np.float32))

    ds_tr = LogMelDataset(train_df, stdz)
    ds_va = LogMelDataset(val_df, stdz)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BetaVAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl"],
        )
        w.writeheader()

    best_val = float("inf")
    bad = 0
    best_path = out_dir / "beta_vae.pt"

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_recon = tr_kl = 0.0
        tr_n = 0

        for xb, _ in dl_tr:
            xb = xb.to(device)
            xhat, mu, logvar = model(xb)
            loss, recon, kl = vae_loss(xb, xhat, mu, logvar, beta=args.beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = xb.size(0)
            tr_loss += float(loss.item()) * bs
            tr_recon += float(recon.item()) * bs
            tr_kl += float(kl.item()) * bs
            tr_n += bs

        tr_loss /= max(tr_n, 1)
        tr_recon /= max(tr_n, 1)
        tr_kl /= max(tr_n, 1)

        # val
        model.eval()
        va_loss = va_recon = va_kl = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, _ in dl_va:
                xb = xb.to(device)
                xhat, mu, logvar = model(xb)
                loss, recon, kl = vae_loss(xb, xhat, mu, logvar, beta=args.beta)

                bs = xb.size(0)
                va_loss += float(loss.item()) * bs
                va_recon += float(recon.item()) * bs
                va_kl += float(kl.item()) * bs
                va_n += bs

        va_loss /= max(va_n, 1)
        va_recon /= max(va_n, 1)
        va_kl /= max(va_n, 1)

        print(
            f"epoch {ep:03d} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_recon={va_recon:.4f} val_kl={va_kl:.4f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl"],
            )
            w.writerow(
                {
                    "epoch": ep,
                    "train_loss": tr_loss,
                    "train_recon": tr_recon,
                    "train_kl": tr_kl,
                    "val_loss": va_loss,
                    "val_recon": va_recon,
                    "val_kl": va_kl,
                }
            )

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    # Load best and export latents for ALL rows of master CSV
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    dl_all = DataLoader(LogMelDataset(master, stdz), batch_size=args.batch_size, shuffle=False, num_workers=0)

    mu_all = []
    with torch.no_grad():
        for xb, _ in dl_all:
            xb = xb.to(device)
            mu, _ = model.enc(xb)
            mu_all.append(mu.cpu().numpy())

    mu_all = np.concatenate(mu_all, axis=0)
    np.save(out_dir / "latent_mu.npy", mu_all)
    print("Saved:", out_dir / "latent_mu.npy", "shape:", mu_all.shape)

    # recon examples
    save_recon_examples(model, ds_va, device, out_dir / "recon_examples.png", n=8)
    print("Saved:", out_dir / "recon_examples.png")


if __name__ == "__main__":
    main()
