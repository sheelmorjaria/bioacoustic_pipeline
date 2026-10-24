"""Trainable causal CTC core. Pretrained full-context features remain offline."""

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from bioacoustic_pipeline.diagnostics.macde_data import frame_observations
from bioacoustic_pipeline.diagnostics.rclat import normalize_text, greedy_text

LABELS = tuple(["-", "|"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["'"])


def encode_text(text):
    symbols = normalize_text(text).replace(" ", "|")
    if not symbols or any(c not in LABELS[1:] for c in symbols):
        raise ValueError("Transcript unsupported by English character alphabet")
    return torch.tensor([LABELS.index(c) for c in symbols], dtype=torch.long)


def acoustic_features(audio, sr):
    audio = np.asarray(audio, float)
    if sr != 16000 or audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) < 640:
        raise ValueError("Finite mono native16kHz audio with one40ms frame required")
    obs = frame_observations(audio, sr)
    frames = np.lib.stride_tricks.sliding_window_view(audio, 640)[::160]
    power = np.abs(np.fft.rfft(frames * np.hanning(640), axis=1)) ** 2
    mel_edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 66) / 2595) - 1)
    hz = np.fft.rfftfreq(640, 1 / sr)
    filters = []
    for a, b, c in zip(mel_edges[:-2], mel_edges[1:-1], mel_edges[2:]):
        triangle = np.maximum(0, np.minimum((hz - a) / (b - a), (c - hz) / (c - b)))
        filters.append(triangle / max(triangle.sum(), 1e-20))
    mel = np.log(power @ np.array(filters).T + 1e-10).astype(np.float32)
    return {
        "mel": mel,
        "mel_valid": np.ones_like(mel, bool),
        "physical": obs["physical"],
        "physical_valid": obs["physical_valid"] & obs["time_valid"][:, None],
        "spectral": obs["spectral"],
        "spectral_valid": obs["spectral_valid"] & obs["time_valid"][:, None],
        "frame_end_samples": obs["frame_end_samples"],
    }


class CausalCTC(nn.Module):
    def __init__(self, arm, input_dim, width=64):
        super().__init__()
        self.arm, self.input_dim, self.width = arm, input_dim, width
        if arm == "combined":
            if input_dim != 76:
                raise ValueError("Combined frontend requires64spectral+12physical features")
            self.spectral = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32))
            self.physical = nn.Sequential(nn.Linear(24, 32), nn.GELU())
            self.frontend = nn.Linear(64, width)
        else:
            self.frontend = nn.Sequential(nn.Linear(2 * input_dim, width), nn.GELU())
        # Decoder initialization is identical across frontend arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.temporal = nn.GRU(width, width, batch_first=True)
            self.head = nn.Linear(width, len(LABELS))

    def forward(self, features, feature_valid, lengths=None, state=None):
        valid = feature_valid.bool() & torch.isfinite(features)
        safe = torch.where(valid, features, 0.0)
        if self.arm == "combined":
            spectral = self.spectral(
                torch.cat([safe[..., :64], valid[..., :64].to(safe.dtype)], -1)
            )
            physical = self.physical(
                torch.cat([safe[..., 64:], valid[..., 64:].to(safe.dtype)], -1)
            )
            inputs = self.frontend(torch.cat([spectral, physical], -1))
        else:
            inputs = self.frontend(torch.cat([safe, valid.to(safe.dtype)], -1))
        if lengths is None:
            output, hidden = self.temporal(inputs, state)
        else:
            packed = pack_padded_sequence(
                inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            encoded, hidden = self.temporal(packed, state)
            output, _ = pad_packed_sequence(
                encoded, batch_first=True, total_length=features.shape[1]
            )
        return self.head(output), hidden


def ctc_objective(logits, lengths, targets):
    if len(targets) != logits.shape[0] or len(lengths) != len(targets):
        raise ValueError("CTC batch schema mismatch")
    for length, target in zip(lengths, targets):
        if len(target) == 0 or (target <= 0).any() or (target >= len(LABELS)).any():
            raise ValueError("Invalid CTC target labels")
        minimum = len(target) + int((target[1:] == target[:-1]).sum())
        if int(length) < minimum or int(length) > logits.shape[1]:
            raise ValueError("Target cannot align to available acoustic frames")
    # XPU CTC backward is nondeterministic. This device transfer is differentiable;
    # neural layers and their gradients remain on XPU while strict CTC runs on CPU.
    ctc_logits = (
        logits.cpu()
        if logits.device.type == "xpu" and torch.are_deterministic_algorithms_enabled()
        else logits
    )
    loss = nn.functional.ctc_loss(
        ctc_logits.log_softmax(-1).transpose(0, 1),
        torch.cat(targets),
        lengths.cpu(),
        torch.tensor([len(t) for t in targets]),
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite CTC training loss")
    return loss


def fit_scaler(sequences):
    count = total = squares = None
    for x, mask in sequences:
        good = np.asarray(mask, bool) & np.isfinite(x)
        safe = np.where(good, x, 0.0).astype(float)
        if count is None:
            count = np.zeros(x.shape[1])
            total = count.copy()
            squares = count.copy()
        count += good.sum(0)
        total += safe.sum(0)
        squares += (safe**2).sum(0)
    if count is None:
        raise ValueError("No training data for scaling")
    mean = total / np.maximum(count, 1)
    variance = np.maximum(0, squares / np.maximum(count, 1) - mean**2)
    return {
        "mean": mean.tolist(),
        "scale": np.where(variance > 1e-10, np.sqrt(variance), 1.0).tolist(),
        "support": (count > 0).tolist(),
    }


def decode_logits(logits, length):
    return greedy_text(logits[: int(length)], LABELS)
