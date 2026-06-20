"""Stimulus builder — exemplar concatenation into playback audio (Phase 1).

Concatenates real per-token segment audio for a token pattern, with a controlled inter-
stimulus interval (ISI) of silence and a short raised-cosine crossfade at each exemplar's
edges to suppress boundary clicks. No neural synthesis — every sample is real segment audio,
so naturalness is bounded by exemplar choice + ISI, not by a vocoder.

This is Phase 1 only. Affect-morphed or fully-synthesized audio (DDSP/vocoder) is Phase 2/3.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from bioacoustic_pipeline.experiments.playback_design.candidate_selector import TokenExemplar


def _fade(n: int) -> np.ndarray:
    """Raised-cosine fade window of length n (0→1 / 1→0 ramp halves)."""
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float32)
    half = n // 2
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, half, dtype=np.float64))
    win = np.concatenate([ramp, np.ones(n - 2 * half), ramp[::-1][: n - half - (n - 2 * half)]])
    return win.astype(np.float32)


def _pick(exemplars: list[TokenExemplar], sr: int, rng: np.random.Generator) -> TokenExemplar:
    """Pick one exemplar, preferring those matching the target sample rate."""
    ok = [e for e in exemplars if e.sr == sr] or exemplars
    return ok[int(rng.integers(0, len(ok)))]


def build_stimulus_audio(
    pattern,
    pool: dict[int, list[TokenExemplar]],
    sr: int,
    *,
    isi_ms: float = 80.0,
    crossfade_ms: float = 6.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Concatenate exemplar audio for ``pattern`` into one stimulus waveform.

    Args:
        pattern: token sequence (ints).
        pool: ``{token: [TokenExemplar, ...]}`` (use ``typical_exemplars`` first).
        sr: target sample rate; exemplars at other rates are still used (no resample).
        isi_ms: inter-token silence gap.
        crossfade_ms: raised-cosine fade at each exemplar's edges.
        seed: RNG seed for deterministic exemplar selection when a token has many.

    Returns ``(audio, provenance)`` where provenance lists the chosen exemplar segment_ids.
    Raises ``KeyError`` if a pattern token has no exemplar in ``pool``.
    """
    rng = np.random.default_rng(seed)
    gap = int(sr * isi_ms / 1000.0)
    fade_n = max(int(sr * crossfade_ms / 1000.0), 1)
    parts: list[np.ndarray] = []
    provenance: list[str] = []
    for i, tok in enumerate(pattern):
        if int(tok) not in pool:
            raise KeyError(f"token {tok} has no exemplar in the pool")
        ex = _pick(pool[int(tok)], sr, rng)
        wav = np.asarray(ex.audio, dtype=np.float32)
        f = _fade(min(fade_n, len(wav)))
        wav = wav.copy()
        wav[: len(f)] *= f
        wav[-len(f) :] *= f[::-1] if len(wav) >= 2 * len(f) else f
        if i > 0:
            parts.append(np.zeros(gap, dtype=np.float32))
        parts.append(wav)
        provenance.append(ex.segment_id or f"tok{tok}")
    if not parts:
        return np.zeros(0, dtype=np.float32), []
    return np.concatenate(parts).astype(np.float32), provenance
