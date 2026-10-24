"""Opt-in phonological soft-reference supervision, separate from frozen ASR models."""

import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from .rclat_simple_ctc import SimpleCTC


class _Readouts(nn.Module):
    def __init__(self, character_head, features):
        super().__init__()
        self.characters = character_head
        self.features = nn.Linear(character_head.in_features, features)

    def forward(self, hidden):
        return self.characters(hidden), self.features(hidden)


class PhonologicalCTC(SimpleCTC):
    """Character and auxiliary logits share continuous encoder states, not hard segments."""

    def __init__(self, feature_names):
        if isinstance(feature_names, str):
            raise ValueError("Feature inventory must be a sequence of feature names")
        names = tuple(feature_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n.strip() for n in names)
        ):
            raise ValueError("A unique, nonempty external feature inventory is required")
        super().__init__()
        self.feature_names = names
        self.head = _Readouts(self.head, len(names))

    def forward(self, features, feature_valid, lengths):
        (characters, phonology), output_lengths = super().forward(features, feature_valid, lengths)
        return characters, phonology, output_lengths


def joint_loss(character_loss, logits, targets, observed, lengths, weight, provenance):
    """Mean over observed feature/frame cells; missing annotations contribute no gradient.

    Targets may be soft probabilities. Their time grid must be independently
    supplied and audited to match encoder outputs; this function never creates
    alignments, interpolates labels or treats acoustic-model posteriors as truth.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Auxiliary weight must be finite and nonnegative")
    if weight == 0:
        return character_loss
    p = provenance or {}
    if (
        p.get("origin") not in {"human_annotation", "independently_validated_reference"}
        or p.get("independent_of_acoustic_model") is not True
        or p.get("source_use_approved") is not True
        or not re.fullmatch("[a-fA-F0-9]{64}", str(p.get("source_sha256", "")))
    ):
        raise ValueError(
            "Active auxiliary supervision requires independent approved reference provenance"
        )
    if (
        logits.ndim != 3
        or logits.shape != targets.shape
        or observed.shape != logits.shape
        or observed.dtype != torch.bool
    ):
        raise ValueError("Feature logits, soft targets and boolean observation mask must match")
    if (
        lengths.dtype not in (torch.int32, torch.int64)
        or lengths.shape != (logits.shape[0],)
        or (lengths <= 0).any()
        or (lengths > logits.shape[1]).any()
    ):
        raise ValueError("Invalid output lengths")
    valid = (
        torch.arange(logits.shape[1], device=logits.device)[None, :, None]
        < lengths.to(logits.device)[:, None, None]
    )
    if bool((observed & ~valid).any()):
        raise ValueError("Reference observations in padding are forbidden")
    if not bool(observed.any()):
        raise ValueError("Active auxiliary loss requires observed reference cells")
    y = targets[observed]
    x = logits[observed]
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()) or bool(
        ((y < 0) | (y > 1)).any()
    ):
        raise ValueError("Observed soft references must be finite probabilities")
    return character_loss + weight * F.binary_cross_entropy_with_logits(x, y)
