#!/usr/bin/env python3
"""
Autoregressive Models for Temporal Context in CPC

Provides two implementations:
1. AutoregressiveMamba: State-space model with O(1) per-step inference
2. TCNAutoregressive: Temporal Convolutional Network (pure PyTorch fallback)

The Mamba model is preferred for its streaming efficiency. The TCN serves
as a fallback when mamba-ssm is not available.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
# See LICENSE file for licensing information.

"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import Mamba
try:
    from mamba_ssm import Mamba as MambaSSM

    MAMBA_AVAILABLE = True
    logger.info("mamba-ssm available: Using Mamba for autoregressive modeling")
except ImportError:
    MAMBA_AVAILABLE = False
    logger.info("mamba-ssm not available: Using TCN fallback")


class Chomp1d(nn.Module):
    """Removes trailing elements from 1D convolution output."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T - chomp_size)
        """
        return x[:, :, : -self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Temporal block for Temporal Convolutional Network."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        def conv1x1(n_in, n_out):
            return nn.Conv1d(n_in, n_out, 1)

        # First causal convolution
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = conv1x1(n_inputs, n_outputs) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for autoregressive modeling.

    Uses dilated convolutions for large receptive field with few parameters.
    Suitable as a fallback when Mamba is unavailable.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: Tuple[int, ...],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation_size,
            )

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) - note: T is second dimension (time)
        Returns:
            (B, T, C)
        """
        # TCN expects (B, C, T), so transpose
        x = x.transpose(1, 2)
        x = self.network(x)
        return x.transpose(1, 2)


class AutoregressiveMamba(nn.Module):
    """
    Autoregressive model using Mamba State Space Model.

    Mamba provides O(1) per-step inference, making it ideal for streaming
    applications with strict latency requirements.

    Falls back to TCN if mamba-ssm is not available.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba: bool = True,
        num_tcn_layers: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_mamba = use_mamba and MAMBA_AVAILABLE

        if self.use_mamba:
            self.mamba = MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            logger.info(f"AutoregressiveMamba: Using Mamba (d_model={d_model})")
        else:
            # Fallback to TCN (configurable depth)
            num_channels = tuple([d_model] * num_tcn_layers)
            self.tcn = TemporalConvNet(
                num_inputs=d_model,
                num_channels=num_channels,
                kernel_size=3,
            )
            logger.info(
                f"AutoregressiveMamba: Using TCN fallback "
                f"(d_model={d_model}, layers={num_tcn_layers})"
            )

    def forward(
        self,
        z_sequence: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process sequence and return context vectors.

        Args:
            z_sequence: (B, T, d_model) latent sequence
            hidden_state: Previous hidden state for streaming (Mamba only)

        Returns:
            context: (B, T, d_model) context vectors
            new_hidden_state: Updated hidden state for next step
        """
        if self.use_mamba:
            # Mamba handles state internally
            context = self.mamba(z_sequence)
            # Note: Mamba doesn't expose hidden state in the same way as RNNs
            # For true streaming, we'd need to use the Mamba state manually
            return context, None
        else:
            # TCN processes full sequence
            context = self.tcn(z_sequence)
            return context, None

    def streaming_step(
        self,
        z_t: torch.Tensor,
        context_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Streaming inference: process single frame with history.

        Args:
            z_t: (B, 1, d_model) current latent frame
            context_history: (B, history_len, d_model) previous context

        Returns:
            c_t: (B, 1, d_model) updated context
        """
        # Concatenate history and current frame
        combined = torch.cat([context_history, z_t], dim=1)

        if self.use_mamba:
            c_t = self.mamba(combined)
        else:
            c_t = self.tcn(combined)

        # Return only the last time step
        return c_t[:, -1:, :]


class TCNAutoregressive(nn.Module):
    """
    Pure PyTorch TCN-based autoregressive model.

    Always available (no external dependencies beyond PyTorch).
    Uses dilated convolutions for efficient temporal modeling.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_tcn_layers: Optional[int] = None,
    ):
        super().__init__()

        # Alias: callers (create_human_cpc_model) pass num_tcn_layers uniformly
        # for both AR backends. AutoregressiveMamba accepts num_tcn_layers
        # directly; accept it here too so the TCN fallback path doesn't crash.
        if num_tcn_layers is not None:
            num_layers = num_tcn_layers

        self.d_model = d_model

        # Create TCN with exponentially increasing dilation
        num_channels = tuple([d_model] * num_layers)
        self.tcn = TemporalConvNet(
            num_inputs=d_model,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Receptive field: each TemporalBlock applies TWO dilated causal convs,
        # so layer l contributes 2*(kernel_size-1)*2**l frames. (The previous
        # formula assumed one conv per block and undercounted ~2x — caught by
        # the §45 streaming-consistency test, which failed at exactly the
        # truncation the wrong bound allowed.)
        self.receptive_field = 1 + sum(
            2 * (kernel_size - 1) * 2**layer for layer in range(num_layers)
        )

        logger.info(
            f"TCNAutoregressive: d_model={d_model}, num_layers={num_layers}, "
            f"receptive_field={self.receptive_field}"
        )

    def forward(self, z_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_sequence: (B, T, d_model)
        Returns:
            context: (B, T, d_model)
        """
        return self.tcn(z_sequence)

    def streaming_step(
        self,
        z_t: torch.Tensor,
        context_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Streaming inference with history buffer.

        Args:
            z_t: (B, 1, d_model) current frame
            context_history: (B, H, d_model) history buffer

        Returns:
            c_t: (B, 1, d_model) current context
        """
        combined = torch.cat([context_history, z_t], dim=1)
        c_all = self.tcn(combined)
        return c_all[:, -1:, :]


class StreamingContextBuffer:
    """
    Maintains a rolling buffer of context for streaming inference.

    Enables O(1) per-step inference by maintaining only the necessary
    history rather than reprocessing the entire sequence.
    """

    def __init__(self, d_model: int, buffer_size: int):
        self.d_model = d_model
        self.buffer_size = buffer_size
        self.buffer: Optional[torch.Tensor] = None

    def update(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Update buffer with new latent frame.

        Args:
            z_t: (B, 1, d_model) or (B, d_model)

        Returns:
            context_history: (B, buffer_size, d_model)
        """
        if z_t.ndim == 2:
            z_t = z_t.unsqueeze(1)

        batch_size = z_t.shape[0]

        # Initialize buffer if needed
        if self.buffer is None:
            self.buffer = torch.zeros(batch_size, self.buffer_size, self.d_model)
            self.buffer = self.buffer.to(z_t.device)

        # Shift buffer and add new frame
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1:, :] = z_t

        return self.buffer

    def get_context(self) -> torch.Tensor:
        """Get current context buffer."""
        return self.buffer

    def reset(self):
        """Reset buffer to zeros."""
        self.buffer = None


def create_autoregressive(
    d_model: int = 128,
    model_type: str = "auto",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create autoregressive model.

    Args:
        d_model: Latent dimension
        model_type: "mamba", "tcn", or "auto" (default)
        **kwargs: Additional model parameters

    Returns:
        Autoregressive model instance
    """
    if model_type == "auto":
        model_type = "mamba" if MAMBA_AVAILABLE else "tcn"

    if model_type == "mamba":
        if not MAMBA_AVAILABLE:
            logger.warning("Mamba requested but unavailable, falling back to TCN")
            return TCNAutoregressive(d_model=d_model, **kwargs)
        return AutoregressiveMamba(d_model=d_model, use_mamba=True, **kwargs)
    elif model_type == "tcn":
        return TCNAutoregressive(d_model=d_model, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test autoregressive models
    batch_size = 4
    seq_len = 32
    d_model = 128

    z = torch.randn(batch_size, seq_len, d_model)

    print("Testing TCN Autoregressive...")
    tcn = TCNAutoregressive(d_model=d_model)
    c_tcn = tcn(z)
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {c_tcn.shape}")
    print(f"Parameters: {sum(p.numel() for p in tcn.parameters()):,}")

    if MAMBA_AVAILABLE:
        print("\nTesting Mamba Autoregressive...")
        mamba = AutoregressiveMamba(d_model=d_model)
        c_mamba, _ = mamba(z)
        print(f"Input shape: {z.shape}")
        print(f"Output shape: {c_mamba.shape}")
        print(f"Parameters: {sum(p.numel() for p in mamba.parameters()):,}")
