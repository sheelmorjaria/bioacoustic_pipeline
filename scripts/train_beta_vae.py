#!/usr/bin/env python3
"""Train the human β-VAE on precomputed CPC latents.

Usage: python scripts/train_beta_vae.py
Trains on /mnt/c/Users/sheel/Desktop/src/data/human_word_features_large.npz
Saves to checkpoints/human_beta_vae/best_model.pt

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bioacoustic_pipeline.embeddings.beta_vae import BetaVAE

NPZ = Path("/mnt/c/Users/sheel/Desktop/src/data/human_word_features_large.npz")
CKPT_OUT = ROOT / "checkpoints" / "human_beta_vae" / "best_model.pt"


def main():
    data = np.load(str(NPZ), allow_pickle=True)
    latents = torch.from_numpy(data["cpc_latents"].astype(np.float32))
    print(f"Loaded {len(latents)} CPC latents, dim={latents.shape[1]}")
    # normalize
    mean = latents.mean(0, keepdim=True)
    std = latents.std(0, keepdim=True) + 1e-6
    latents = (latents - mean) / std

    ds = TensorDataset(latents)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    model = BetaVAE(input_dim=latents.shape[1], latent_dim=16, beta=4.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float("inf")
    for epoch in range(50):
        model.train()
        total = 0
        for (batch,) in loader:
            recon, mu, logvar, z = model(batch)
            loss, rl, kl = model.loss(batch, recon, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(ds)
        if avg < best_loss:
            best_loss = avg
            CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"input_dim": latents.shape[1], "latent_dim": 16,
                           "hidden_dim": 128, "beta": 4.0},
                "norm_mean": mean.numpy(), "norm_std": std.numpy(),
                "best_loss": best_loss,
            }, CKPT_OUT)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}: loss={avg:.1f} (recon={rl:.1f} kl={kl:.1f}) best={best_loss:.1f}")
    print(f"Done. Best loss={best_loss:.1f} -> {CKPT_OUT}")


if __name__ == "__main__":
    main()
