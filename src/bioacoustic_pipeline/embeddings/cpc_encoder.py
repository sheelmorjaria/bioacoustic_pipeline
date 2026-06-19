#!/usr/bin/env python3
"""
CPC Encoder: 1D Convolutional Encoder for Audio Frames

Encodes raw audio frames into latent representations z_t for Contrastive
Predictive Coding. Uses strided convolutions for sub-sampling and temporal
resolution reduction.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
# See LICENSE file for licensing information.

"""

import logging
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """Configuration for CPC Encoder."""

    sample_rate: int = 48000  # Hz
    frame_size_ms: int = 10  # Frame size in milliseconds
    hidden_dim: int = 128  # Latent dimension z_t
    num_channels: Tuple[int, ...] = (64, 128, 256)
    kernel_sizes: Tuple[int, ...] = (5, 5, 3)
    strides: Tuple[int, ...] = (2, 2, 1)
    dropout: float = 0.1

    @property
    def frame_size_samples(self) -> int:
        """Frame size in samples."""
        return int(self.sample_rate * self.frame_size_ms / 1000)


class CPCEncoder(nn.Module):
    """
    Encodes raw audio frames into latent space z_t.

    Architecture:
    - 3 layers of strided 1D convolutions
    - GELU activations
    - Optional dropout for regularization
    - Output: (Batch, Time_steps, hidden_dim)

    Example:
        >>> encoder = CPCEncoder(sample_rate=48000, frame_size_ms=10)
        >>> audio = torch.randn(4, 1, 480)  # 4 batches, 10ms @ 48kHz
        >>> z = encoder(audio)  # Shape: (4, T, 128)
    """

    def __init__(
        self,
        sample_rate: Union[int, "EncoderConfig"] = 48000,
        frame_size_ms: int = 10,
        hidden_dim: int = 128,
        num_channels: Tuple[int, ...] = (64, 128, 256),
        kernel_sizes: Tuple[int, ...] = (5, 5, 3),
        strides: Tuple[int, ...] = (2, 2, 1),
        dropout: float = 0.1,
    ):
        super().__init__()

        # Accept an EncoderConfig as the first positional argument.
        if isinstance(sample_rate, EncoderConfig):
            cfg = sample_rate
            sample_rate = cfg.sample_rate
            frame_size_ms = cfg.frame_size_ms
            hidden_dim = cfg.hidden_dim
            num_channels = cfg.num_channels
            kernel_sizes = cfg.kernel_sizes
            strides = cfg.strides
            dropout = cfg.dropout

        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.frame_size_samples = int(sample_rate * frame_size_ms / 1000)
        self.hidden_dim = hidden_dim

        # Validate configuration
        assert len(num_channels) == len(kernel_sizes) == len(strides), (
            "num_channels, kernel_sizes, and strides must have same length"
        )

        # Build encoder layers
        layers = []
        in_channels = 1  # Mono audio

        for i, (out_ch, kernel, stride) in enumerate(zip(num_channels, kernel_sizes, strides)):
            # Conv1d layer
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_ch,
                        kernel_size=kernel,
                        stride=stride,
                        padding=kernel // 2,  # Maintain temporal alignment
                    ),
                    nn.GELU(),
                ]
            )

            if dropout > 0 and i < len(num_channels) - 1:
                layers.append(nn.Dropout(dropout))

            in_channels = out_ch

        # Final projection to hidden_dim
        layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=1))

        self.encoder = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"CPCEncoder initialized: sample_rate={sample_rate}Hz, "
            f"frame_size={frame_size_ms}ms ({self.frame_size_samples} samples), "
            f"hidden_dim={hidden_dim}"
        )

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to latent representation.

        Args:
            x: Input audio tensor (Batch, 1, Time_steps)

        Returns:
            Latent representation (Batch, T, hidden_dim)
        """
        # x: (B, 1, T)
        z = self.encoder(x)  # (B, hidden_dim, T')
        z = z.transpose(1, 2)  # (B, T', hidden_dim)
        return z

    def encode_frame(self, audio: np.ndarray) -> np.ndarray:
        """
        Encode a single audio frame (numpy interface for compatibility).

        Args:
            audio: Audio samples (frame_size_samples,) or (1, frame_size_samples)

        Returns:
            Latent representation (hidden_dim,)
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        assert audio.shape[1] == self.frame_size_samples, (
            f"Expected {self.frame_size_samples} samples, got {audio.shape[1]}"
        )

        with torch.no_grad():
            x = torch.from_numpy(audio).float().unsqueeze(1)  # (1, 1, T)
            z = self.forward(x)  # (1, T', hidden_dim)
            # Take the mean across time for single frame representation
            z_mean = z.mean(dim=1).squeeze().cpu().numpy()

        return z_mean

    @property
    def receptive_field_ms(self) -> float:
        """Calculate receptive field in milliseconds."""
        # Simplified calculation (actual depends on strides)
        total_stride = 1
        for stride in [2, 2, 1]:  # Hardcoded for default config
            total_stride *= stride

        return (self.frame_size_samples / total_stride) / self.sample_rate * 1000

    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output length after convolutions.

        Args:
            input_length: Input sequence length

        Returns:
            Output sequence length
        """
        length = input_length
        for stride in [2, 2, 1]:  # Default strides
            length = (length + stride - 1) // stride
        return length


class LightweightCPCEncoder(nn.Module):
    """
    Lightweight encoder for edge deployment.

    Reduced parameters and computation for Jetson Orin Nano compatibility.
    Maintains similar representational capacity with fewer channels.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_size_ms: int = 10,
        hidden_dim: int = 64,  # Smaller latent dimension
        num_channels: Tuple[int, ...] = (32, 64, 128),  # Fewer channels
        kernel_sizes: Tuple[int, ...] = (5, 5, 3),
        strides: Tuple[int, ...] = (2, 2, 1),
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.frame_size_samples = int(sample_rate * frame_size_ms / 1000)
        self.hidden_dim = hidden_dim

        layers = []
        in_channels = 1

        for out_ch, kernel, stride in zip(num_channels, kernel_sizes, strides):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_ch, kernel, stride, padding=kernel // 2),
                    nn.GELU(),
                ]
            )
            in_channels = out_ch

        layers.append(nn.Conv1d(in_channels, hidden_dim, 1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z.transpose(1, 2)

    @property
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_encoder(
    config: EncoderConfig,
    lightweight: bool = False,
) -> CPCEncoder:
    """
    Factory function to create encoder from configuration.

    Args:
        config: Encoder configuration
        lightweight: If True, use LightweightCPCEncoder for edge deployment

    Returns:
        Encoder instance
    """
    if lightweight:
        return LightweightCPCEncoder(
            sample_rate=config.sample_rate,
            frame_size_ms=config.frame_size_ms,
            hidden_dim=config.hidden_dim,
        )
    else:
        return CPCEncoder(
            sample_rate=config.sample_rate,
            frame_size_ms=config.frame_size_ms,
            hidden_dim=config.hidden_dim,
            num_channels=config.num_channels,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test encoder
    encoder = CPCEncoder()

    # Generate test audio
    batch_size = 4
    audio = torch.randn(batch_size, 1, encoder.frame_size_samples)

    # Encode
    z = encoder(audio)
    print(f"Input shape: {audio.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Receptive field: {encoder.receptive_field_ms:.2f}ms")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
