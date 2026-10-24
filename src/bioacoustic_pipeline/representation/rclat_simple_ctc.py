"""Small convolutional–bidirectional LSTM CTC trainability control."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .rclat_ctc import LABELS
from .rclat_scale_v4 import valid_frames


class SimpleCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(128, 128)
        self.subsample = nn.ModuleList(
            [nn.Conv1d(128, 128, 3, stride=2, padding=1, bias=False) for _ in range(2)]
        )
        self.recurrent = nn.LSTM(128, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(256, len(LABELS))

    def forward(self, features, feature_valid, lengths):
        if features.ndim != 3 or features.shape != feature_valid.shape or features.shape[2] != 64:
            raise ValueError("Feature schema mismatch")
        if len(lengths) != len(features) or (lengths <= 0).any() or (lengths > features.shape[1]).any():
            raise ValueError("Invalid native lengths")
        valid = valid_frames(lengths, features.shape[1], features.device)
        supported = feature_valid.bool() & torch.isfinite(features) & valid[..., None]
        safe = torch.where(supported, features, 0.0)
        x = self.projection(torch.cat([safe, supported.to(features.dtype)], -1))
        x = x.masked_fill(~valid[..., None], 0.0)
        output_lengths = lengths.clone()
        for convolution in self.subsample:
            x = nn.functional.silu(convolution(x.transpose(1, 2))).transpose(1, 2)
            output_lengths = (output_lengths + 1) // 2
            valid = valid_frames(output_lengths, x.shape[1], x.device)
            x = x.masked_fill(~valid[..., None], 0.0)
        # Separate valid-length sequences prevent backward recurrence seeing padding.
        sequences = [self.recurrent(row[None, :int(n)])[0][0] for row, n in zip(x, output_lengths)]
        return self.head(pad_sequence(sequences, batch_first=True)), output_lengths
