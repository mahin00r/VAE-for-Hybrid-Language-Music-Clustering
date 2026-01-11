import os
import random
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------- Paths ----------
DATA_CSV = "data/processed/dataset_clean.csv"
X_NPY = "data/processed/audio_features_clean.npy"
SPLIT_DIR = "data/processed/splits"

OUT_DIR = "data/processed/results_vae"
MODEL_PATH = os.path.join(OUT_DIR, "vae.pt")
SCALER_PATH = os.path.join(OUT_DIR, "scaler.joblib")
LATENT_PATH = os.path.join(OUT_DIR, "latent_mu.npy")
TRAIN_LOG = os.path.join(OUT_DIR, "train_log.csv")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NpyDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i]


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar


def vae_loss(x, xhat, mu, logvar, beta=1.0):
    recon = torch.mean((xhat - x) ** 2)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon.item(), kld.item()


def load_split_indices(df, split_name):
    split_path = os.path.join(SPLIT_DIR, f"{split_name}.csv")
    split_df = pd.read_csv(split_path)

    id_to_idx = {sid: i for i, sid in enumerate(df["segment_id"].astype(str).tolist())}
    idx = [id_to_idx[sid] for sid in split_df["segment_id"].astype(str).tolist() if sid in id_to_idx]
    return np.array(idx, dtype=int)


@torch.no_grad()
def extract_mu(model, X, device, batch_size=256):
    model.eval()
    loader = DataLoader(NpyDataset(X), batch_size=batch_size, shuffle=False)
    mus = []
    for xb in loader:
        xb = xb.to(device)
        h = model.enc(xb)
        mu = model.mu(h)
        mus.append(mu.cpu().numpy())
    return np.vstack(mus)


def main(
    latent_dim=16,
    hidden=128,
    lr=1e-3,
    batch_size=128,
    epochs=80,
    beta=1.0,
    seed=42,
    patience=10,
):
    ensure_dir(OUT_DIR)
    set_seed(seed)

    df = pd.read_csv(DATA_CSV)
    X = np.load(X_NPY).astype(np.float32)

    train_idx = load_split_indices(df, "train")
    val_idx = load_split_indices(df, "val")

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Train/val split is empty or does not match dataset_clean.csv")

    # scale on TRAIN only
    scaler = StandardScaler().fit(X[train_idx])
    Xs = scaler.transform(X).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = VAE(input_dim=X.shape[1], latent_dim=latent_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(NpyDataset(Xs[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NpyDataset(Xs[val_idx]), batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    bad = 0
    logs = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for xb in train_loader:
            xb = xb.to(device)
            xhat, mu, logvar = model(xb)
            loss, recon, kld = vae_loss(xb, xhat, mu, logvar, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)

        tr_loss /= max(1, tr_n)

        # validation
        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device)
                xhat, mu, logvar = model(xb)
                loss, recon, kld = vae_loss(xb, xhat, mu, logvar, beta=beta)
                va_loss += loss.item() * xb.size(0)
                va_n += xb.size(0)

        va_loss /= max(1, va_n)

        logs.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"epoch {ep:03d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            bad = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    pd.DataFrame(logs).to_csv(TRAIN_LOG, index=False)
    print("Saved log:", TRAIN_LOG)
    print("Saved model:", MODEL_PATH)
    print("Saved scaler:", SCALER_PATH)

    # export latent means for all samples
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    mu_all = extract_mu(model, Xs, device=device)
    np.save(LATENT_PATH, mu_all)
    print("Saved latent:", LATENT_PATH, "shape:", mu_all.shape)


if __name__ == "__main__":
    main()
