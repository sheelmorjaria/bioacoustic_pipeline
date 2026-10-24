"""Vocal-target CPC: eligibility masking for the two-arm pilot.

Changes WHICH predictions drive learning — not the architecture, codebook,
caller routing, or negative policy. Target eligibility comes from the Phase-1
activity detector's validity masks; anchors with no eligible negatives are
excluded (NaN), never reassuring zeros. Genuine gaps are preserved in their
original temporal positions; padding is inert.

The two arms share: starting checkpoint, optimizer state, additional training
budget. They differ only in the frozen vocal-target selection rule. Compare
on the same held-out source examples, eligible targets, and candidate
identities — lower training InfoNCE alone proves nothing (different target
mixes change task difficulty).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import torch


def eligible_target_mask(vocal_frames: torch.Tensor, *, steps_ahead: int) -> torch.Tensor:
    """Per-batch count of eligible (t, k) prediction targets.

    A target (t, k) is eligible when frame t is vocal (the anchor's local
    context is vocal material) — the simple first-ablation rule. Fancier
    rules (target-frame vocal, context availability) are separate ablations.
    """
    vocal = torch.as_tensor(vocal_frames, dtype=torch.bool)
    if vocal.ndim > 1:
        n_frames = vocal.shape[1]
        return vocal.sum(dim=1).clamp(max=n_frames - steps_ahead)
    # 1-D = per-item eligibility (True: the item's frames are vocal); the
    # caller supplies n_frames context for the count
    raise ValueError("pass per-frame vocal masks (batch, T), not per-item flags")


def vocal_mask_from_audio(
    audio: np.ndarray, sr: int, *, frame_ms: float = 10.0, threshold_db: float = -36.0
) -> np.ndarray:
    """Per-frame vocal mask from log-RMS above threshold (dB re FS)."""
    x = np.asarray(audio, dtype=np.float64)
    frame = max(int(sr * frame_ms / 1000.0), 1)
    n = len(x) // frame
    blocks = x[: n * frame].reshape(n, frame)
    rms = np.sqrt((blocks**2).mean(axis=1))
    levels = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return levels >= threshold_db


def masked_cpc_loss(
    model,
    audio_frames: torch.Tensor,
    vocal_frames: torch.Tensor,
    *,
    temperature: float = 0.07,
    n_negatives: int | None = None,
) -> torch.Tensor:
    """CPC loss averaged over VOCAL-eligible anchors only.

    Rows (batch items) with zero eligible targets are excluded entirely —
    their loss contributes NaN, not zero, so a fully-silent batch reports
    'no signal' rather than a reassuring 0.0.
    """
    vocal = torch.as_tensor(vocal_frames, dtype=torch.bool)
    eligible_rows = (vocal.sum(dim=1) > 0) if vocal.ndim > 1 else vocal
    if not bool(eligible_rows.any()):
        return torch.tensor(float("nan"))
    audio_eligible = audio_frames[eligible_rows]
    z, c, p = model(audio_eligible)
    # parent loss consumes len(predictions) for its horizon loop
    return model.compute_loss(
        z, c, [None] * model.steps_ahead, temperature, n_negatives=n_negatives
    )


def two_arm_pilot_spec() -> dict:
    """Prespecified comparison contract (fixed before results are seen)."""
    return {
        "arms": {
            "reference": "existing corrected objective on valid observed audio; "
            "no artificial-padding admission",
            "vocal_target": "same model/setup; additional frozen vocal-target selection rule",
        },
        "shared": [
            "starting checkpoint",
            "optimizer state",
            "additional training budget (updates, not epochs)",
        ],
        "endpoints": [
            "held-out discrimination of actual vocal continuations against the "
            "same alternatives (real-history and controlled-history reported "
            "separately)",
            "recurrence recovery of independently source-supported matches "
            "(NOT raw low-distance pair counts)",
        ],
        "explicitly_not_endpoints": [
            "training InfoNCE (different target mixes change task difficulty)",
            "making T_y positive (a silence-aware population model is useful "
            "even if T_y remains ~0)",
        ],
        "provenance": "pilot checkpoints carry parent-checkpoint hash and "
        "loss-history; never 'trained from scratch on vocal data'",
        "framing": "controlled training ablation — T0 computational, NOT meaning",
    }


def fixed_bank_loss(model, audio_frames, vocal_frames, *, bank_idx,
                    temperature: float = 0.07) -> torch.Tensor:
    """Loss with SOURCE-INDEXED candidates — padding-inertness instrument.

    ``bank_idx`` indexes the latent pool of the VALID prefix; appending
    padding cannot add candidates because the bank is fixed to valid rows.
    """
    vocal = torch.as_tensor(vocal_frames, dtype=torch.bool)
    eligible = (vocal.sum(dim=1) > 0) if vocal.ndim > 1 else vocal
    if not bool(eligible.any()):
        return torch.tensor(float("nan"))
    z, c, _ = model(audio_frames[eligible])
    z_flat = torch.nn.functional.normalize(z.reshape(-1, z.shape[-1]), dim=1)
    bank = z_flat[bank_idx]
    # single-step InfoNCE on the fixed bank
    anchors = torch.nn.functional.normalize(c.reshape(-1, c.shape[-1]), dim=1)
    pos = z_flat.roll(-1, dims=0)  # next-frame targets (source-indexed)
    logits = torch.cat([
        (anchors * pos).sum(-1, keepdim=True),
        anchors @ bank.T,
    ], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long,
                         device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def per_target_vocal_loss(model, audio_frames, vocal_frames, *,
                          temperature: float = 0.07,
                          n_negatives: int = 5) -> torch.Tensor:
    """Horizon-aligned per-target vocal loss on the PRESERVED timeline.

    Representations are computed on the full sequence (no packing); eligible
    prediction instances are (b, t, k) where the TARGET frame t+k is vocal
    and the context frame t is observed. Reduction: mean over eligible
    instances across horizons (aggregation documented; not per-horizon mean
    of means).
    """
    import torch.nn.functional as F  # noqa: PLC0415

    vocal = torch.as_tensor(vocal_frames, dtype=torch.bool)
    z, c, _ = model(audio_frames)  # FULL timeline forward — no packing
    B, T, H = z.shape
    K = model.steps_ahead
    # eligible: context frame t valid (any), target frame t+k vocal
    # (assume all observed frames have valid context per the first-ablation
    # rule; per-frame context validity is a separate ablation)
    losses = []
    for k in range(1, K + 1):
        anchors = c[:, : T - k]  # (B, T-k, H) — context at t
        positives = z[:, k:]  # (B, T-k, H) — target at t+k
        mask = vocal[:, k:]  # (B, T-k) — target frame vocal
        if not bool(mask.any()):
            continue
        a_norm = F.normalize(anchors, dim=2)
        p_norm = F.normalize(positives, dim=2)
        # in-batch negatives from the full pool (observed timeline only)
        pool = F.normalize(z.reshape(-1, H), dim=1)
        neg_idx = torch.randint(0, pool.shape[0], (n_negatives,),
                                device=z.device)
        neg_bank = pool[neg_idx]
        pos_logit = (a_norm * p_norm).sum(-1) / temperature  # (B, T-k)
        neg_logits = torch.einsum("bth,kh->btk", a_norm, neg_bank) / temperature
        logits = torch.cat(
            [pos_logit.unsqueeze(-1), neg_logits], dim=-1
        ).reshape(-1, 1 + n_negatives)
        labels = torch.zeros(logits.shape[0], dtype=torch.long,
                             device=z.device)
        per_anchor = F.cross_entropy(logits, labels, reduction="none")
        per_anchor = per_anchor.reshape(B, T - k)
        eligible = mask.reshape(-1)
        selected = per_anchor.reshape(-1)[eligible]
        if selected.numel() > 0:
            losses.append(selected)
    if not losses:
        return torch.tensor(float("nan"))
    return torch.cat(losses).mean()  # mean over ALL eligible instances


def safe_vocal_train_step(model, optimizer, audio_frames, vocal_frames, *,
                          temperature: float = 0.07,
                          n_negatives: int = 5) -> dict:
    """Empty-safe vocal training step.

    no_eligible_targets -> NO backward, NO optimizer.step(), NO state
    mutation, NO scheduler advance (returned dict carries the status; the
    caller counts attempted vs completed updates separately).
    """
    vocal = torch.as_tensor(vocal_frames, dtype=torch.bool)
    eligible = (vocal.sum(dim=1) > 0) if vocal.ndim > 1 else vocal
    if not bool(eligible.any()):
        return {"status": "no_eligible_targets", "completed_update": False,
                "loss": None}
    loss = per_target_vocal_loss(model, audio_frames, vocal_frames,
                                 temperature=temperature,
                                 n_negatives=n_negatives)
    if not torch.isfinite(loss):
        return {"status": "nonfinite_loss", "completed_update": False,
                "loss": None}
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"status": "trained", "completed_update": True,
            "loss": float(loss)}
