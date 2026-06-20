"""β-VAE for the affective stream — disentangled continuous latent from CPC.

The affective dual-stream complement to the VQ-VAE syntactic token: where the VQ-VAE
collapses to discrete acoustic-similarity classes (the Semantic Ceiling), the β-VAE
learns a continuous, disentangled affective/prosodic latent (β>1 promotes factor
separation). Input is the CPC 256-D latent; output is a 16-D affect vector.

Trained on human CPC latents from LibriSpeech (human_word_features_large.npz).
Human-only (firewall: applies_to_species=["human"]).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BetaVAE(nn.Module):
    """β-VAE: encoder (input→μ,logvar→z) + decoder (z→reconstruction)."""

    def __init__(self, input_dim: int = 256, latent_dim: int = 16,
                 hidden_dim: int = 128, beta: float = 4.0):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim
        # encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        # decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss(self, x, recon, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl, recon_loss.item(), kl.item()

    @torch.no_grad()
    def affect_vector(self, x):
        """Inference: CPC latent → 16-D affect vector (posterior mean)."""
        self.eval()
        mu, _ = self.encode(x.unsqueeze(0) if x.dim() == 1 else x)
        return mu.squeeze(0)
