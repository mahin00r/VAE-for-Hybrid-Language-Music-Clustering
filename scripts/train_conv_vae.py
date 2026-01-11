import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _to_2d(a: np.ndarray) -> np.ndarray:
    """Force log-mel to shape (F, T)."""
    a = np.asarray(a)
    if a.ndim == 3:
        # common: (1, F, T) or (F, T, 1)
        if a.shape[0] == 1:
            a = a[0]
        elif a.shape[-1] == 1:
            a = a[..., 0]
        else:
            a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Expected 2D logmel, got shape={a.shape}")

    # Heuristic: if freq dimension looks like time (very large) and time dimension looks like mel bins (<=256), transpose
    if a.shape[0] > a.shape[1] and a.shape[1] <= 256:
        a = a.T
    return a.astype(np.float32)


def crop_or_pad_2d(a: np.ndarray, F: int, T: int) -> np.ndarray:
    """Return array of shape (F, T) by cropping or zero-padding."""
    out = np.zeros((F, T), dtype=np.float32)
    f = min(F, a.shape[0])
    t = min(T, a.shape[1])
    out[:f, :t] = a[:f, :t]
    return out


class LogMelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, indices: np.ndarray, target_shape: tuple[int, int]):
        self.df = df.reset_index(drop=True)
        self.indices = indices.astype(int)
        self.F, self.T = target_shape

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.df.iloc[self.indices[i]]
        path = row["logmel_path"]
        a = np.load(path)
        a = _to_2d(a)
        a = crop_or_pad_2d(a, self.F, self.T)

        # per-sample normalization (simple and robust)
        m = float(a.mean())
        s = float(a.std() + 1e-6)
        a = (a - m) / s

        x = torch.from_numpy(a).unsqueeze(0)  # (1, F, T)
        return x


class ConvVAE(nn.Module):
    def __init__(self, input_shape: tuple[int, int], latent_dim: int = 16, base_ch: int = 32):
        super().__init__()
        F, T = input_shape

        self.enc = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, F, T)
            h = self.enc(dummy)
            self._enc_shape = h.shape[1:]  # (C, f, t)
            flat_dim = int(np.prod(self._enc_shape))

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_ch, 1, 3, stride=2, padding=1, output_padding=1),
        )

        self.input_shape = input_shape

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), *self._enc_shape)
        xhat = self.dec(h)

        # crop/pad decoder output to exact input shape
        F, T = self.input_shape
        xhat = xhat[:, :, :F, :T]
        if xhat.shape[2] < F or xhat.shape[3] < T:
            out = torch.zeros(xhat.size(0), 1, F, T, device=xhat.device)
            out[:, :, :xhat.shape[2], :xhat.shape[3]] = xhat
            xhat = out
        return xhat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar


def vae_loss(x, xhat, mu, logvar, beta: float = 1.0):
    recon = torch.mean((xhat - x) ** 2)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon.item(), kld.item()


def load_split_indices(df: pd.DataFrame, split_csv: str) -> np.ndarray:
    s = pd.read_csv(split_csv)
    id_to_idx = {sid: i for i, sid in enumerate(df["segment_id"].astype(str).tolist())}
    idx = [id_to_idx[sid] for sid in s["segment_id"].astype(str).tolist() if sid in id_to_idx]
    return np.array(idx, dtype=int)


@torch.no_grad()
def extract_mu_all(model: ConvVAE, df: pd.DataFrame, target_shape: tuple[int, int], device: str, batch_size: int):
    model.eval()
    idx = np.arange(len(df), dtype=int)
    ds = LogMelDataset(df, idx, target_shape)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    mus = []
    for xb in dl:
        xb = xb.to(device)
        mu, _ = model.encode(xb)
        mus.append(mu.cpu().numpy())
    return np.vstack(mus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/processed_medium")
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)

    in_dir = args.in_dir
    df = pd.read_csv(os.path.join(in_dir, "dataset_clean.csv"))

    # Determine target logmel shape from first file
    first = np.load(df.iloc[0]["logmel_path"])
    first = _to_2d(first)
    target_shape = (first.shape[0], first.shape[1])
    print("Target logmel shape (F,T):", target_shape)

    # splits
    tr_idx = load_split_indices(df, os.path.join(in_dir, "splits", "train.csv"))
    va_idx = load_split_indices(df, os.path.join(in_dir, "splits", "val.csv"))

    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise RuntimeError("Train/val split ended up empty. Check your splits files.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = ConvVAE(input_shape=target_shape, latent_dim=args.latent_dim, base_ch=args.base_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dl = DataLoader(LogMelDataset(df, tr_idx, target_shape), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(LogMelDataset(df, va_idx, target_shape), batch_size=args.batch_size, shuffle=False)

    out_dir = os.path.join(in_dir, "results_conv_vae")
    ensure_dir(out_dir)
    model_path = os.path.join(out_dir, "conv_vae.pt")
    log_path = os.path.join(out_dir, "train_log.csv")
    latent_path = os.path.join(out_dir, "latent_mu.npy")

    best_val = float("inf")
    bad = 0
    logs = []

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for xb in train_dl:
            xb = xb.to(device)
            xhat, mu, logvar = model(xb)
            loss, recon, kld = vae_loss(xb, xhat, mu, logvar, beta=args.beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)

        tr_loss /= max(1, tr_n)

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for xb in val_dl:
                xb = xb.to(device)
                xhat, mu, logvar = model(xb)
                loss, recon, kld = vae_loss(xb, xhat, mu, logvar, beta=args.beta)
                va_loss += loss.item() * xb.size(0)
                va_n += xb.size(0)

        va_loss /= max(1, va_n)
        logs.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"epoch {ep:03d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            bad = 0
            torch.save(model.state_dict(), model_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    pd.DataFrame(logs).to_csv(log_path, index=False)
    print("Saved log:", log_path)
    print("Saved model:", model_path)

    # Export latent means for all samples (in df order)
    model.load_state_dict(torch.load(model_path, map_location=device))
    mu_all = extract_mu_all(model, df, target_shape, device=device, batch_size=args.batch_size)
    np.save(latent_path, mu_all)
    print("Saved latent:", latent_path, "shape:", mu_all.shape)


if __name__ == "__main__":
    main()
