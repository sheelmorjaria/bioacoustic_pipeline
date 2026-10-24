"""Loss utilities: valid-anchor-reduced cross entropy (§46 gate).

An InfoNCE anchor whose negatives are all masked (e.g. every sampled
negative collided with its exact positive) has softmax mass 1 on the
positive — per-anchor loss 0. A plain ``mean`` lets those zeros dilute the
objective; in per-frame EVALUATION they would read as falsely "perfect"
predictions. ``masked_cross_entropy`` divides by the count of anchors that
still have at least one finite negative.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over (N, C) logits, mean over VALID anchors only.

    Column 0 is the positive (label 0); anchors are valid when at least one
    negative column (1:) is finite. No valid anchors -> 0.0 (never NaN).
    """
    per_anchor = F.cross_entropy(logits, labels, reduction="none")
    valid = torch.isfinite(logits[:, 1:]).any(dim=1).to(per_anchor.dtype)
    return (per_anchor * valid).sum() / valid.sum().clamp(min=1.0)
