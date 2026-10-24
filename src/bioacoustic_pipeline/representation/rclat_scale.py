"""Offline Conformer-style CTC encoder for new scaling diagnostics.

Frame-wise LayerNorm replaces time-aggregating convolution normalisation so padded
frames cannot change a shorter utterance. This is not a streaming recogniser.
"""

import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .rclat_ctc import LABELS


def valid_frames(lengths, total, device):
    return torch.arange(total, device=device)[None] < lengths.to(device)[:, None]


class Block(nn.Module):
    def __init__(self, width, heads, expansion, kernel, dropout):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion, width),
            nn.Dropout(dropout),
        )
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.convolution_norm = nn.LayerNorm(width)
        self.point_in = nn.Conv1d(width, 2 * width, 1, bias=False)
        self.depth = nn.Conv1d(width, width, kernel, groups=width, padding=kernel // 2, bias=False)
        self.frame_norm = nn.LayerNorm(width)
        self.point_out = nn.Conv1d(width, width, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion, width),
            nn.Dropout(dropout),
        )
        self.final = nn.LayerNorm(width)

    def forward(self, x, valid):
        mask = valid[..., None]
        x = (x + 0.5 * self.ff1(x)).masked_fill(~mask, 0.0)
        q = self.attention_norm(x)
        with sdpa_kernel(SDPBackend.MATH):
            attended, _ = self.attention(q, q, q, key_padding_mask=~valid, need_weights=False)
        x = (x + self.drop(attended)).masked_fill(~mask, 0.0)
        y = self.point_in(self.convolution_norm(x).transpose(1, 2))
        y = nn.functional.glu(y, dim=1).masked_fill(~valid[:, None], 0.0)
        y = self.depth(y).transpose(1, 2)
        y = nn.functional.silu(self.frame_norm(y)).masked_fill(~mask, 0.0)
        y = self.point_out(y.transpose(1, 2)).transpose(1, 2)
        x = (x + self.drop(y)).masked_fill(~mask, 0.0)
        return self.final(x + 0.5 * self.ff2(x)).masked_fill(~mask, 0.0)


class ScaleCTC(nn.Module):
    def __init__(
        self, input_dim, width=128, layers=4, heads=4, expansion=512, kernel=15, dropout=0.0
    ):
        super().__init__()
        if width % 2 or width % heads or kernel % 2 != 1:
            raise ValueError("Even width divisible by heads and odd convolution kernel required")
        self.input_dim = input_dim
        self.projection = nn.Linear(2 * input_dim, width)
        # Common encoder and output initialization regardless of frontend dimension.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.subsample = nn.ModuleList(
                [nn.Conv1d(width, width, 3, stride=2, padding=1, bias=False) for _ in range(2)]
            )
            self.blocks = nn.ModuleList(
                [Block(width, heads, expansion, kernel, dropout) for _ in range(layers)]
            )
            self.head = nn.Linear(width, len(LABELS))
        self.register_buffer(
            "position_frequencies",
            torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width)),
        )
        self.width = width

    def forward(self, features, feature_valid, lengths):
        if (
            features.ndim != 3
            or features.shape != feature_valid.shape
            or features.shape[2] != self.input_dim
        ):
            raise ValueError("Feature schema mismatch")
        if (
            len(lengths) != len(features)
            or (lengths <= 0).any()
            or (lengths > features.shape[1]).any()
        ):
            raise ValueError("Invalid native frame lengths")
        valid = valid_frames(lengths, features.shape[1], features.device)
        supported = feature_valid.bool() & torch.isfinite(features) & valid[..., None]
        safe = torch.where(supported, features, 0.0)
        x = self.projection(torch.cat([safe, supported.to(features.dtype)], -1)).masked_fill(
            ~valid[..., None], 0.0
        )
        out_lengths = lengths.clone()
        for convolution in self.subsample:
            x = nn.functional.silu(convolution(x.transpose(1, 2))).transpose(1, 2)
            out_lengths = (out_lengths + 1) // 2
            valid = valid_frames(out_lengths, x.shape[1], x.device)
            x = x.masked_fill(~valid[..., None], 0.0)
        angles = (
            torch.arange(x.shape[1], device=x.device)[:, None] * self.position_frequencies[None]
        )
        positions = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        x = (x + positions[None]).masked_fill(~valid[..., None], 0.0)
        for block in self.blocks:
            x = block(x, valid)
        return self.head(x), out_lengths
