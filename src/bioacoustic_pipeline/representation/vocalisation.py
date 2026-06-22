"""Hierarchical dual-stream vocalisation token — the canonical call representation.

Each vocalisation is ONE structured event, not a flat token:

    DualStreamVocalisation (parent = whole call)
      ├ discrete stream:   vq_token · motif_token · fingerprint_cluster · alias_role
      ├ continuous stream: affect_mu · acoustic_fingerprint · confidence
      └ sub_segments:      [DualStreamSubToken, ...]   (HECB internal structure)

The parent captures the whole call; the children capture typed internal parts. Default
downstream representation is **whole-call primary + sub-call augmentation** (pooled), NOT
sub-call replacement — empirically validated: pooled > whole > sub-call-alone across
MeerKAT + InfantMarmosetsVox (call-type + caller-ID).

Cross-species guardrail: discrete vq_tokens are SPECIES-SPECIFIC (do not compare IDs across
species). Cross-species comparison uses the shared interpretable features
(`acoustic_fingerprint` = physics + macro + micro) + structural summaries (subcall_count,
relative boundary positions) — never raw species token IDs.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from bioacoustic_pipeline.segmentation.hierarchical import CallEvent
from bioacoustic_pipeline.representation.subcall import (
    build_subcall_representations,
    pooled_representation,
)


@dataclass
class DualStreamSubToken:
    """One typed internal sub-call part of a DualStreamVocalisation."""

    parent_vocalisation_id: str = ""
    start_s: float = 0.0
    end_s: float = 0.0
    relative_start: float = 0.0  # (start - parent_onset) / parent_duration
    relative_end: float = 0.0
    boundary_type: str = "internal"  # onset_body | internal | offset_body
    positional_role: str = ""  # onset | body | offset | only (derived from position in parent)
    # discrete stream
    vq_token: int = -1
    fingerprint_cluster: int | None = None
    # continuous stream
    affect_mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    acoustic_fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict:
        return {
            "relative_start": round(self.relative_start, 4),
            "relative_end": round(self.relative_end, 4),
            "boundary_type": self.boundary_type,
            "positional_role": self.positional_role,
            "fingerprint_cluster": self.fingerprint_cluster,
        }


@dataclass
class DualStreamVocalisation:
    """One vocalisation as a hierarchical dual-stream token (parent whole call)."""

    vocalisation_id: str = ""
    file_id: str = ""
    species: str = ""
    caller_id: str | None = None
    calltype: str | None = None
    start_s: float = 0.0
    end_s: float = 0.0
    # --- discrete / symbolic stream ---
    vq_token: int = -1
    motif_token: str | None = None
    fingerprint_cluster: int | None = None
    alias_role: str | None = None  # functional-alias role (context-associated, NOT meaning)
    # --- continuous / acoustic stream ---
    affect_mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    acoustic_fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(0))
    confidence: float = 1.0
    # --- hierarchy ---
    sub_segments: list[DualStreamSubToken] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def subcall_count(self) -> int:
        return len(self.sub_segments)

    @property
    def relative_boundary_positions(self) -> list[float]:
        """Normalized positions of internal boundaries within the call (0=start, 1=end)."""
        dur = self.duration_s
        if dur <= 0 or self.subcall_count == 0:
            return []
        # boundaries = the cuts between sub-segments (exclude the 0.0 and 1.0 edges)
        return sorted(
            {round(s.relative_end, 4) for s in self.sub_segments if 0 < s.relative_end < 1}
        )

    def structural_summary(self) -> dict:
        """Cross-species structural descriptors (species-agnostic — no token IDs)."""
        n = self.subcall_count
        rel = self.relative_boundary_positions
        return {
            "duration_s": round(self.duration_s, 4),
            "subcall_count": n,
            "relative_boundary_positions": rel,
            "mean_internal_gap": round(float(np.mean(np.diff([0.0] + rel + [1.0]))), 4)
            if rel
            else 1.0,
        }


def _relative(t_s: float, onset_s: float, dur_s: float) -> float:
    return (t_s - onset_s) / dur_s if dur_s > 0 else 0.0


def build_vocalisation(
    call: CallEvent,
    audio: np.ndarray,
    sr: int,
    *,
    fingerprint_fn: Callable[[np.ndarray, int], np.ndarray],
    tokenizer=None,
    affect_backbone=None,
    species: str = "",
    calltype: str | None = None,
    caller_id: str | None = None,
    file_id: str = "",
) -> DualStreamVocalisation | None:
    """CallEvent (+ internal_segments) → DualStreamVocalisation (whole-call parent + sub-children).

    Returns None if the whole call is too short / silent (no whole fingerprint).
    """
    # Reuse the proven subcall builder for the whole + sub fingerprints + pooled vector.
    reps = build_subcall_representations([call], audio, sr, fingerprint_fn)
    if not reps:
        return None
    rep = reps[0]
    dur = call.duration_ms / 1000.0
    voc = DualStreamVocalisation(
        vocalisation_id=call.call_id,
        file_id=file_id or call.file_id,
        species=species,
        caller_id=caller_id,
        calltype=calltype,
        start_s=call.onset_s,
        end_s=call.offset_s,
        acoustic_fingerprint=rep.whole_fingerprint,
        confidence=float(call.crnn_confidence),
    )
    # discrete streams (optional backbones)
    i = int(call.onset_s * sr)
    j = max(int(call.offset_s * sr), i + int(0.03 * sr))
    whole_audio = audio[i:j]
    if tokenizer is not None:
        voc.vq_token = tokenizer.tokenize(whole_audio, sr)
    if affect_backbone is not None and voc.vq_token != -1:
        try:
            voc.affect_mu = np.asarray(affect_backbone.embed(whole_audio, sr), dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass
    # sub-call children
    n_subs = len(call.internal_segments)
    for si, seg in enumerate(call.internal_segments):
        si_sample = int(seg.start_s * sr)
        sj_sample = max(int(seg.end_s * sr), si_sample + int(0.012 * sr))
        sseg = audio[si_sample:sj_sample]
        if n_subs <= 1:
            role = "only"
        elif si == 0:
            role = "onset"
        elif si == n_subs - 1:
            role = "offset"
        else:
            role = "body"
        sub = DualStreamSubToken(
            parent_vocalisation_id=voc.vocalisation_id,
            start_s=seg.start_s,
            end_s=seg.end_s,
            relative_start=_relative(seg.start_s, call.onset_s, dur),
            relative_end=_relative(seg.end_s, call.onset_s, dur),
            boundary_type=seg.boundary_type or "internal",
            positional_role=role,
        )
        if len(sseg) >= int(0.012 * sr):
            sfp = np.asarray(fingerprint_fn(sseg, sr), dtype=np.float32)
            if np.any(sfp):
                sub.acoustic_fingerprint = sfp
                if tokenizer is not None:
                    sub.vq_token = tokenizer.tokenize(sseg, sr)
        voc.sub_segments.append(sub)
    return voc


# ─── 4-way ablation: whole / sub-call / pooled / whole+sequence ─────────────


def _kmeans_codebook(vecs: np.ndarray, k: int, seed: int = 0):
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(vecs)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaler.transform(vecs))
    return scaler, km.cluster_centers_


def _tokenize(vecs, scaler, cents):
    sv = scaler.transform(vecs)
    return np.argmin(((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1)


def _sequence_vector(
    voc: DualStreamVocalisation, fp_dim: int, k: int, sub_scaler=None, sub_cents=None
) -> np.ndarray:
    """whole || sub-call cluster histogram || [n_sub, mean_relative_gap].

    The histogram captures the sub-call SEQUENCE composition (which internal parts recur);
    without a fitted codebook (sub_scaler=None) it falls back to the pooled mean. Fixed
    dimension across calls (whole + k histogram bins + 2) so it clusters/probes cleanly."""
    whole = voc.acoustic_fingerprint
    hist = np.zeros(k, dtype=np.float32)
    gap = np.zeros(2, dtype=np.float32)
    sfps = [s.acoustic_fingerprint for s in voc.sub_segments if s.acoustic_fingerprint.size]
    if sfps and sub_scaler is not None and sub_cents is not None:
        toks = _tokenize(np.asarray(sfps, dtype=np.float32), sub_scaler, sub_cents)
        for t in toks:
            hist[int(t)] += 1.0
        n = float(len(sfps))
        hist = hist / max(n, 1.0)
        gap[0] = n
        rel = voc.relative_boundary_positions
        gap[1] = float(np.mean(np.diff([0.0] + rel + [1.0]))) if rel else 1.0
    elif sfps:
        # fallback: mean sub-call fp instead of histogram
        hist = np.mean(np.asarray(sfps, dtype=np.float32), axis=0)[:k]  # truncated to k
        gap[0] = float(len(sfps))
    return np.concatenate([whole, hist, gap]).astype(np.float32)


def whole_sub_pooled_sequence_ablation(
    vocs: list[DualStreamVocalisation],
    labels: list,
    *,
    k: int = 12,
) -> dict:
    """4-way ablation: whole / sub-call / pooled / whole+relative-sequence → label recovery.

    The 'sequence' representation (whole ‖ sub-call cluster histogram ‖ [n_sub, gap]) tests
    whether the ORDERED internal composition carries signal beyond the pooled mean.
    """
    from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415
    from bioacoustic_pipeline.diagnostics.probes import linear_probe  # noqa: PLC0415

    if len(vocs) != len(labels):
        raise ValueError(f"vocs ({len(vocs)}) and labels ({len(labels)}) must align")
    labels_arr = np.asarray(labels)
    whole = np.asarray([v.acoustic_fingerprint for v in vocs], dtype=np.float32)
    pooled = np.asarray(
        [
            pooled_representation(
                v.acoustic_fingerprint,
                [s.acoustic_fingerprint for s in v.sub_segments if s.acoustic_fingerprint.size],
            )
            for v in vocs
        ],
        dtype=np.float32,
    )
    # sub-call codebook (fit once on all sub-fps for the sequence histogram)
    all_sub = [
        s.acoustic_fingerprint for v in vocs for s in v.sub_segments if s.acoustic_fingerprint.size
    ]
    sub_scaler = sub_cents = None
    if len(all_sub) >= k * 5:
        sub_scaler, sub_cents = _kmeans_codebook(np.asarray(all_sub, dtype=np.float32), k)
    fp_dim = whole.shape[1] if whole.size else 0
    sequence = np.asarray(
        [_sequence_vector(v, fp_dim, k, sub_scaler, sub_cents) for v in vocs], dtype=np.float32
    )

    def _rec(vecs, lbls=labels_arr):
        if vecs.shape[0] < 2 or vecs.shape[1] == 0:
            return {"V": 0.0, "nmi": 0.0, "p": None, "probe": None}
        scaler, cents = _kmeans_codebook(vecs, k)
        toks = _tokenize(vecs, scaler, cents)
        b = association_test(
            [[int(t)] for t in toks],
            np.asarray(lbls).tolist(),
            statistic="nmi",
            n_permutations=200,
            seed=0,
        ).as_dict()
        probe = linear_probe(vecs, lbls, min_per_class=5).as_dict()
        return {
            "V": b["cramers_v"],
            "nmi": b["observed_nmi"],
            "p": b["p_value"],
            "probe": probe.get("accuracy"),
        }

    result = {
        "n_vocs": len(vocs),
        "n_with_internal": sum(1 for v in vocs if v.subcall_count > 0),
        "whole": _rec(whole),
        "pooled": _rec(pooled),
        "sequence": _rec(sequence),
        "subcall": None,
    }
    if len(all_sub) >= k * 5:
        sub_labels = [
            lab
            for v, lab in zip(vocs, labels)
            for s in v.sub_segments
            if s.acoustic_fingerprint.size
        ]
        result["subcall"] = _rec(np.asarray(all_sub, dtype=np.float32), sub_labels)
    return result


def format_4way_table(result: dict) -> str:
    lines = [
        f"vocalisations: {result['n_vocs']} ({result['n_with_internal']} with internal)",
        f"{'representation':28} {'Cramers V':>10} {'NMI':>8} {'p':>10} {'probe':>8}",
        "-" * 68,
    ]

    def _row(name, r):
        if r is None:
            return f"{name:28} {'(n/a)':>10}"
        return f"{name:28} {r['V']:10.3f} {r['nmi']:8.3f} {str(r['p']):>10} {str(r['probe']):>8}"

    lines.append(_row("whole call", result["whole"]))
    lines.append(_row("whole + sub pool", result["pooled"]))
    lines.append(_row("whole + sub SEQUENCE", result["sequence"]))
    lines.append(_row("sub-call (parent lbl)", result["subcall"]))
    return "\n".join(lines)
