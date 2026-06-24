"""DSP Audio Splicer — turn HECB-segmented calls into counterfactual WAV stimuli.

Track C (Architecture v1.0). Given a CallEvent with internal_segments (each carrying absolute
start_s/end_s) + the source audio, this module slices the per-unit audio and produces
falsifiable counterfactual stimuli for H0-CMC playback experiments:

  natural            — the original call audio (unchanged baseline)
  order_shuffled     — permute the unit order (destroy sequence syntax, preserve units)
  onset_offset_swap  — swap the first and last units (destroy positional roles)
  caller_swapped     — replace one unit with a donor from another caller (test identity)
  amplitude_matched  — RMS-normalise to a target (acoustic-intensity control)

These are exemplar-based DSP transforms (slice + reorder + concatenate), NOT neural synthesis —
deterministic, firewall-clean, and transparent for the wet lab. The existing
`experiments.playback_design` infra (write_audio, preflight_stimulus, build_manifest) handles
output + naturalness screening.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class SpliceResult:
    """One counterfactual stimulus."""

    condition: (
        str  # natural | order_shuffled | onset_offset_swap | caller_swapped | amplitude_matched
    )
    audio: np.ndarray
    sr: int
    n_units: int
    provenance: dict = field(
        default_factory=dict
    )  # transform details (permutation, donor idx, etc.)


def extract_unit_audio(
    call_onset_s: float,
    call_offset_s: float,
    segments: Sequence[tuple[float, float]],
    audio: np.ndarray,
    sr: int,
    min_samples: int = 192,
) -> list[np.ndarray]:
    """Slice the call's internal segments → list of per-unit audio arrays.

    Args:
        call_onset_s / call_offset_s: the parent call's absolute onset/offset (seconds).
        segments: [(start_s, end_s), ...] absolute seconds for each internal unit.
        audio: the full-clip audio array.
        sr: sample rate.
        min_samples: discard units shorter than this (avoid clicks).
    """
    units = []
    for s, e in segments:
        si = max(int(s * sr), 0)
        sj = max(int(e * sr), si + min_samples)
        seg = audio[si:sj]
        if len(seg) >= min_samples and np.any(seg):
            units.append(np.asarray(seg, dtype=np.float32))
    return units


def _apply_crossfade(units: list[np.ndarray], fade_ms: float = 2.0, sr: int = 16000) -> np.ndarray:
    """Concatenate units with a short crossfade to avoid click artifacts at boundaries."""
    if not units:
        return np.zeros(0, dtype=np.float32)
    if len(units) == 1:
        return units[0]
    fade = int(fade_ms / 1000 * sr)
    out = list(units)
    for i in range(1, len(out)):
        fa = min(fade, len(out[i - 1]) // 2, len(out[i]) // 2)
        if fa > 0:
            ramp = np.linspace(0, 1, fa, dtype=np.float32)
            out[i - 1] = out[i - 1].copy()
            out[i - 1][-fa:] *= 1 - ramp
            out[i] = out[i].copy()
            out[i][:fa] *= ramp
    return np.concatenate(out)


def splice_natural(units: list[np.ndarray], sr: int) -> SpliceResult:
    """The original call (units in their natural order)."""
    return SpliceResult(
        "natural", _apply_crossfade(units, sr=sr), sr, len(units), {"order": "natural"}
    )


def splice_order_shuffled(units: list[np.ndarray], sr: int, rng: random.Random) -> SpliceResult:
    """Permute the unit order (destroy sequence syntax; preserve acoustic units)."""
    if len(units) < 2:
        return splice_natural(units, sr)
    perm = list(range(len(units)))
    rng.shuffle(perm)
    shuffled = [units[i] for i in perm]
    return SpliceResult(
        "order_shuffled", _apply_crossfade(shuffled, sr=sr), sr, len(units), {"permutation": perm}
    )


def splice_onset_offset_swap(units: list[np.ndarray], sr: int) -> SpliceResult:
    """Swap the first and last units (destroy positional roles; preserve units)."""
    if len(units) < 2:
        return splice_natural(units, sr)
    swapped = list(units)
    swapped[0], swapped[-1] = swapped[-1], swapped[0]
    return SpliceResult(
        "onset_offset_swap",
        _apply_crossfade(swapped, sr=sr),
        sr,
        len(units),
        {"swapped": (0, len(units) - 1)},
    )


def splice_caller_swapped(
    units: list[np.ndarray], sr: int, donor_pool: list[np.ndarray], rng: random.Random
) -> SpliceResult:
    """Replace one unit with a donor from another caller's call (test identity invariance).

    The donor should be from a DIFFERENT caller (same species, different individual). The
    position is random; the donor is a random unit from the pool.
    """
    if len(units) < 2 or not donor_pool:
        return splice_natural(units, sr)
    idx = rng.randrange(len(units))
    donor = rng.choice(donor_pool)
    out = list(units)
    out[idx] = donor
    return SpliceResult(
        "caller_swapped",
        _apply_crossfade(out, sr=sr),
        sr,
        len(units),
        {"replaced_unit": idx},
    )


def splice_amplitude_matched(
    units: list[np.ndarray], sr: int, target_rms: float = 0.05
) -> SpliceResult:
    """RMS-normalise the concatenated call to a target amplitude (acoustic-intensity control).

    This is a CONTROL stimulus: same units, same order, same duration — just amplitude-scaled.
    It rules out the "the animal responded to loudness, not structure" confound.
    """
    audio = _apply_crossfade(units, sr=sr)
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-12)
    scaled = audio * (target_rms / rms)
    return SpliceResult(
        "amplitude_matched",
        scaled.astype(np.float32),
        sr,
        len(units),
        {"target_rms": target_rms, "original_rms": rms},
    )


def generate_all_counterfactuals(
    units: list[np.ndarray],
    sr: int,
    donor_pool: list[np.ndarray] | None = None,
    seed: int = 0,
) -> list[SpliceResult]:
    """Generate the full H0-CMC counterfactual set for one call.

    Returns: [natural, order_shuffled, onset_offset_swap, caller_swapped, amplitude_matched].
    If donor_pool is empty, caller_swapped falls back to natural.
    """
    rng = random.Random(seed)
    results = [
        splice_natural(units, sr),
        splice_order_shuffled(units, sr, rng),
        splice_onset_offset_swap(units, sr),
        splice_caller_swapped(units, sr, donor_pool or [], rng),
        splice_amplitude_matched(units, sr),
    ]
    return results


def splice_isolated_unit(units: list[np.ndarray], sr: int, idx: int = 0) -> SpliceResult:
    """Render a single unit in isolation (compositional playback: Isolated A / Isolated B).

    Same DSP path as the in-sequence rendering (crossfade is a no-op on one unit) so the
    isolated unit is acoustically comparable to its rendering inside AB. This is the control
    that asks: "is the receiver's response to the AB *sequence* greater than its response to
    either part alone?" — the structural test for compositionality (T2), never a meaning claim.
    """
    if not units:
        return SpliceResult(
            "isolated_unit", np.zeros(0, dtype=np.float32), sr, 0, {"isolated_index": idx}
        )
    j = idx if 0 <= idx < len(units) else 0
    return SpliceResult(
        "isolated_unit", _apply_crossfade([units[j]], sr=sr), sr, 1, {"isolated_index": j}
    )


def splice_reversed(units: list[np.ndarray], sr: int) -> SpliceResult:
    """Reverse the unit order — the canonical "same parts, wrong order" control (Shuffled BA).

    Distinct from ``splice_order_shuffled`` (a *random* permutation): reversed is the specific
    AB -> BA transform that tests whether order/positional roles carry the effect. For a 2-unit
    candidate it is exactly the BA counterfactual.
    """
    if len(units) < 2:
        return splice_natural(units, sr)
    rev = list(units)[::-1]
    return SpliceResult(
        "reversed", _apply_crossfade(rev, sr=sr), sr, len(units), {"order": "reversed"}
    )


def generate_compositional_counterfactuals(
    units: list[np.ndarray], sr: int, a_idx: int = 0, b_idx: int = 1
) -> list[SpliceResult]:
    """Compositional (H0-CMC) counterfactual set for a candidate AB sequence.

    Returns: [natural_AB, isolated_A, isolated_B, reversed_BA, amplitude_matched_AB].

    Design: if AB is compositional, the receiver's response to natural_AB should exceed the
    response to isolated_A and isolated_B (greater-than-the-sum) and differ from reversed_BA
    (order matters). amplitude_matched controls for loudness. This is a T2 stimulus set; the
    wet lab decides whether AB is perceived as compositional. The pipeline asserts NOTHING
    about meaning (semantic ceiling intact).
    """
    if not units:
        return [splice_natural(units, sr)]
    a = min(a_idx, len(units) - 1)
    b = min(b_idx, len(units) - 1)
    return [
        splice_natural(units, sr),
        splice_isolated_unit(units, sr, a),
        splice_isolated_unit(units, sr, b),
        splice_reversed(units, sr),
        splice_amplitude_matched(units, sr),
    ]
