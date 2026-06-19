#!/usr/bin/env python3
"""
VQ-VAE for Syntactic Encoding (Stream 2)

Implements a VQ-VAE with EMA codebook updates for discrete syntactic tokenization.
Encodes 44D syntactic features to discrete tokens (0-63).

Key Benefits:
- EMA codebook updates prevent codebook collapse
- >80% codebook utilization target
- Codebook revival for dead tokens

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
# See LICENSE file for licensing information.

"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VQVAEConfig:
    """Configuration for VQ-VAE."""

    input_dim: int = 61  # Must match frozen ONNX encoder (1, 61)
    codebook_size: int = 64
    codebook_dim: int = 32
    hidden_dim: int = 128
    commitment_cost: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5
    # Revival threshold: tokens used less than this % of batches are "dead"
    revival_threshold: float = 0.01


class EMAVectorQuantizer(nn.Module):
    """
    Vector quantizer with EMA codebook updates and codebook revival.

    Prevents codebook collapse via:
    1. Exponential moving average of encoder outputs
    2. Codebook revival for unused tokens
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        revival_threshold: float = 0.01,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.epsilon = epsilon
        self.revival_threshold = revival_threshold

        # EMA codebook
        self.register_buffer(
            "codebook_ema",
            torch.randn(codebook_size, codebook_dim),
        )
        self.register_buffer(
            "cluster_size_ema",
            torch.zeros(codebook_size),
        )

        # Track per-token usage for revival (alias for compatibility)
        self.register_buffer(
            "token_usage_count",
            torch.zeros(codebook_size),
        )
        # Alias for tests that expect usage_count
        self.usage_count = self.token_usage_count

        self.register_buffer(
            "total_batches",
            torch.tensor(0.0),
        )

    def update_ema(self, z_flat: torch.Tensor, token_ids: torch.Tensor) -> None:
        """
        Update EMA codebook and cluster sizes.

        This is called during training to move the codebook vectors
        toward the mean of encoder outputs assigned to each code.
        """
        # Get one-hot encoding of token assignments
        encodings = F.one_hot(token_ids, self.codebook_size).float()

        # Sum over batch to get counts per code
        cluster_sizes = encodings.sum(dim=0)

        # Weighted sum of encoder outputs for each code
        # z_flat: (batch * seq, codebook_dim)
        # encodings.T: (codebook_size, batch * seq)
        # encodings.T @ z_flat: (codebook_size, codebook_dim)
        encoded_sum = torch.matmul(encodings.T, z_flat)

        # Update EMA cluster sizes
        self.cluster_size_ema.mul_(self.decay).add_(cluster_sizes, alpha=1 - self.decay)

        # Update EMA codebook
        # Add small epsilon to avoid division by zero
        cluster_size_ema_expanded = self.cluster_size_ema.unsqueeze(1)
        codebook_update = encoded_sum / (cluster_size_ema_expanded + self.epsilon)
        self.codebook_ema.mul_(self.decay).add_(codebook_update, alpha=1 - self.decay)

        # Track usage for revival
        self.token_usage_count.add_((cluster_sizes > 0).float())
        self.total_batches.add_(1.0)

    def revive_dead_codes(
        self, z_flat: Optional[torch.Tensor] = None, threshold: Optional[float] = None
    ) -> None:
        """
        Revive dead codes by replacing them with random encoder outputs.

        Args:
            z_flat: Flattened encoder outputs (optional for test compatibility)
            threshold: Optional override for revival threshold (for test compatibility)
        """
        if self.total_batches < 10:
            # Don't revive in the first few batches
            return

        # Use provided threshold or default to revival_threshold
        rev_threshold = threshold if threshold is not None else self.revival_threshold

        # Calculate usage percentage for each code
        usage_rate = self.token_usage_count / self.total_batches

        # Find dead codes (usage rate below threshold)
        dead_codes = usage_rate < rev_threshold

        if dead_codes.any():
            num_dead = dead_codes.sum().item()
            if num_dead > 0:
                # If z_flat is provided, use it to sample replacement vectors
                if z_flat is not None:
                    num_samples = min(num_dead, z_flat.shape[0])
                    indices = torch.randperm(z_flat.shape[0], device=z_flat.device)[:num_samples]

                    # Replace dead codes with random encoder outputs
                    dead_indices = torch.where(dead_codes)[0]
                    for i, dead_idx in enumerate(dead_indices[:num_samples]):
                        self.codebook_ema[dead_idx] = z_flat[indices[i]].detach()
                        self.cluster_size_ema[dead_idx] = 0.0
                        self.token_usage_count[dead_idx] = 0.0
                else:
                    # Just reset usage counts for dead codes (for test compatibility)
                    dead_indices = torch.where(dead_codes)[0]
                    for dead_idx in dead_indices:
                        self.cluster_size_ema[dead_idx] = 0.0
                        self.token_usage_count[dead_idx] = 0.0

                logger.debug(f"Revived {num_dead} dead codes")

    def forward(self, z: torch.Tensor, training: bool = True) -> tuple:
        """
        Quantize z using EMA codebook.

        Args:
            z: Encoder output of shape (batch, ..., codebook_dim)
            training: Whether in training mode (updates EMA, revives dead codes)

        Returns:
            z_q: Quantized vectors
            token_ids: Discrete token indices
            perplexity: Codebook utilization metric
        """
        # Flatten for distance computation
        z_flat = z.reshape(-1, self.codebook_dim)

        # Compute distances to codebook
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z^T * e
        z_norm = (z_flat**2).sum(dim=1, keepdim=True)
        codebook_norm = (self.codebook_ema**2).sum(dim=1)
        distances = z_norm + codebook_norm - 2 * torch.matmul(z_flat, self.codebook_ema.t())

        # Find nearest codebook entry
        token_ids = torch.argmin(distances, dim=1)

        # Reshape to original shape
        token_ids = token_ids.view(z.shape[:-1])

        # Quantize
        z_flat_q = self.codebook_ema[token_ids.flatten()]
        z_q = z_flat_q.view_as(z)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Track token usage even during inference (for utilization stats)
        encodings = F.one_hot(token_ids.flatten(), self.codebook_size).float()
        self.token_usage_count.add_((encodings.sum(0) > 0).float())

        # Update EMA and revive dead codes during training
        if training:
            self.total_batches.add_(1.0)
            self.update_ema(z_flat, token_ids.flatten())
            self.revive_dead_codes(z_flat)

        # Perplexity for monitoring
        perplexity = torch.exp(
            -torch.sum(
                encodings.mean(0) * torch.log(encodings.mean(0) + self.epsilon),
            )
        )

        return z_q, token_ids, perplexity

    def get_utilization_stats(self) -> dict:
        """Get codebook utilization statistics."""
        # Check if we have any usage at all
        has_usage = self.token_usage_count.sum() > 0

        if not has_usage:
            # Return zeros before any forward pass
            return {
                "utilization_percent": 0.0,
                "active_codes": 0,
                "total_codes": self.codebook_size,
                "dead_codes": self.codebook_size,
                "per_code_usage": [0.0] * self.codebook_size,
            }

        # Calculate utilization based on raw usage counts
        # Any code with usage_count > 0 is considered active
        active_codes = (self.token_usage_count > 0).sum().item()
        utilization_percent = active_codes / self.codebook_size * 100

        # For per-code usage, normalize by max usage to get relative frequencies
        max_usage = self.token_usage_count.max().item()
        if max_usage > 0:
            per_code_usage = (self.token_usage_count / max_usage).cpu().numpy().tolist()
        else:
            per_code_usage = [0.0] * self.codebook_size

        return {
            "utilization_percent": utilization_percent,
            "active_codes": active_codes,
            "total_codes": self.codebook_size,
            "dead_codes": self.codebook_size - active_codes,
            "per_code_usage": per_code_usage,
        }

    def get_codebook_utilization(self) -> float:
        """Get codebook utilization as fraction (0-1)."""
        stats = self.get_utilization_stats()
        return stats["utilization_percent"] / 100.0


class SyntacticVQVAE(nn.Module):
    """VQ-VAE: ~44D syntactic features → N discrete tokens."""

    def __init__(
        self,
        input_dim: int = 61,  # Must match frozen ONNX encoder (1, 61)
        codebook_size: int = 64,
        codebook_dim: int = 32,
        hidden_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        revival_threshold: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, codebook_dim),
        )

        # VQ layer with EMA and revival
        self.vq = EMAVectorQuantizer(
            codebook_size,
            codebook_dim,
            decay,
            revival_threshold=revival_threshold,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode from quantized latent."""
        return self.decoder(z_q)

    def decode_code(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Decode directly from token IDs."""
        # Get codebook entries for the token IDs
        z_q = self.vq.codebook_ema[token_ids]
        # Decode the quantized vectors
        return self.decode(z_q)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple:
        """Forward pass: encode → quantize → decode."""
        z = self.encode(x)
        z_q, token_ids, perplexity = self.vq(z, training=self.training)
        x_recon = self.decode(z_q)
        return x_recon, z_q, token_ids, perplexity

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        z_q: torch.Tensor,
    ) -> tuple:
        """VQ-VAE loss with commitment term.

        Returns:
            Tuple of (total_loss, recon_loss, commit_loss)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Commitment loss (encoder commitment to codebook)
        commit_loss = F.mse_loss(z_q.detach(), z)

        # Note: codebook loss is handled via EMA updates, not backprop
        total_loss = recon_loss + commit_loss * self.commitment_cost

        return total_loss, recon_loss, commit_loss

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize input to discrete tokens (inference)."""
        with torch.no_grad():
            z = self.encode(x)
            _, token_ids, _ = self.vq(z, training=False)
        return token_ids

    def codebook_utilization(self) -> float:
        """Calculate codebook utilization percentage."""
        stats = self.vq.get_utilization_stats()
        return stats["utilization_percent"]

    def get_codebook_utilization(self) -> float:
        """Get codebook utilization as fraction (0-1)."""
        return self.codebook_utilization() / 100.0

    def get_utilization_stats(self) -> dict:
        """Get detailed codebook utilization statistics."""
        return self.vq.get_utilization_stats()


def create_syntactic_vqvae(config: Optional[VQVAEConfig] = None) -> SyntacticVQVAE:
    """Create a VQ-VAE with default or custom config."""
    if config is None:
        config = VQVAEConfig()

    return SyntacticVQVAE(
        input_dim=config.input_dim,
        codebook_size=config.codebook_size,
        codebook_dim=config.codebook_dim,
        hidden_dim=config.hidden_dim,
        commitment_cost=config.commitment_cost,
        decay=config.decay,
        revival_threshold=config.revival_threshold,
    )


class VQVAECheckpoint:
    """Checkpoint management for VQ-VAE models."""

    def __init__(self, model: SyntacticVQVAE, config: Optional[VQVAEConfig] = None):
        """Initialize checkpoint manager.

        Args:
            model: The VQ-VAE model to checkpoint
            config: Optional configuration to save with checkpoint
        """
        self.model = model
        self.config = config or VQVAEConfig()

    def save(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_dim": self.model.input_dim
                    if hasattr(self.model, "input_dim")
                    else self.config.input_dim,
                    "codebook_size": self.model.codebook_size,
                    "codebook_dim": self.model.codebook_dim,
                    "commitment_cost": self.model.commitment_cost,
                },
            },
            path,
        )

    @classmethod
    def save_checkpoint(
        cls,
        model: SyntacticVQVAE,
        optimizer,
        epoch: int,
        loss: float,
        path: Path,
    ) -> None:
        """Save a full training checkpoint (model + optimizer + metadata)."""
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
                "config": {
                    "input_dim": getattr(model, "input_dim", 44),
                    "codebook_size": model.codebook_size,
                    "codebook_dim": model.codebook_dim,
                    "commitment_cost": model.commitment_cost,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "VQVAECheckpoint":
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            VQVAECheckpoint instance with loaded model
        """
        data = torch.load(path)
        config_data = data.get("config", {})

        config = VQVAEConfig(
            input_dim=config_data.get("input_dim", 44),
            codebook_size=config_data.get("codebook_size", 64),
            codebook_dim=config_data.get("codebook_dim", 32),
            commitment_cost=config_data.get("commitment_cost", 0.25),
        )

        model = create_syntactic_vqvae(config)
        model.load_state_dict(data["state_dict"])

        return cls(model, config)
