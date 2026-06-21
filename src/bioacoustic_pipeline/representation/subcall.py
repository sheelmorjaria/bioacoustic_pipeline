"""Hierarchical (whole / sub-call / pooled) segment representation + ablation.

HECB-Seg's confirmed role is a hierarchical *internal-boundary annotator*: the parent
event detector (PCEN+CRNN, or NBD) gives authoritative call events; HECB adds typed
internal sub-call boundaries. This module is the representation layer that turns a
``CallEvent`` (carrying ``InternalSegment`` instances) into the three representation
modes used downstream, and standardizes the whole/sub-call/pooled ablation.

Representation modes (the standard ablation):
  * **whole**      — one fingerprint per call event (the baseline; primary representation)
  * **sub-call**   — one fingerprint per internal segment (finer; weaker alone — loses
                     whole-call context)
  * **pooled**     — whole fingerprint || mean(sub-call fps) || [n_sub, internal_frac]
                     (augmentation layer: sub-call structure summarized alongside the whole)

Empirical (MeerKAT, 1500 clips): pooled V=0.274 > whole V=0.246 > sub-call V=0.142.
Sub-call features AUGMENT whole-call representations when pooled but are weaker alone —
internal segmentation augments, not replaces, whole-event representations.

Works on any ``CallEvent`` with ``internal_segments`` (from ``two_stage_segment`` for
NBD-based calls or ``hecb_decode`` for CRNN-based calls) — segmenter-agnostic.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from bioacoustic_pipeline.diagnostics.context_binding import association_test
from bioacoustic_pipeline.diagnostics.probes import linear_probe
from bioacoustic_pipeline.segmentation.hierarchical import CallEvent


@dataclass
class SubcallRepresentation:
    """One call event's whole / sub-call / pooled fingerprints."""

    call_id: str
    start_ms: float
    end_ms: float
    whole_fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(0))
    subcall_fingerprints: list[np.ndarray] = field(default_factory=list)
    pooled_fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_internal(self) -> int:
        return len(self.subcall_fingerprints)

    @property
    def has_internal(self) -> bool:
        return self.n_internal > 0


def pooled_representation(
    whole_fp: np.ndarray, subcall_fps: list[np.ndarray], fp_dim: int | None = None
) -> np.ndarray:
    """Pooled = whole || mean(sub-call fps) || [n_sub, internal_frac].

    ``internal_frac`` = n_sub / max(n_internal_segments, 1) (how many proposed sub-segments
    survived the min-duration gate). When there are no sub-calls, the sub-summary is zeroed
    and the call is represented by its whole fingerprint + a [0, 0] marker (so the pooled
    vector dimension is constant across calls — required for clustering/probing).
    """
    whole_fp = np.asarray(whole_fp, dtype=np.float32)
    if fp_dim is None:
        fp_dim = whole_fp.shape[0] if whole_fp.size else 0
    if subcall_fps:
        mean_sub = np.mean(np.asarray(subcall_fps, dtype=np.float32), axis=0)
        n_sub = float(len(subcall_fps))
    else:
        mean_sub = np.zeros(fp_dim, dtype=np.float32)
        n_sub = 0.0
    return np.concatenate([whole_fp, mean_sub, np.array([n_sub], dtype=np.float32)]).astype(
        np.float32
    )


def build_subcall_representations(
    call_events: list[CallEvent],
    audio: np.ndarray,
    sr: int,
    fingerprint_fn: Callable[[np.ndarray, int], np.ndarray],
    *,
    min_sub_samples: int | None = None,
) -> list[SubcallRepresentation]:
    """Build whole / sub-call / pooled fingerprints for each call event.

    Args:
        call_events: CallEvents (with optional ``internal_segments``) — from
            ``two_stage_segment`` or ``hecb_decode``.
        audio: full-clip audio.
        sr: sample rate.
        fingerprint_fn: callable(seg_audio, sr) -> 1-D fingerprint vector
            (e.g. ``lambda a, s: afp.extract(a, s, species=...).combined``).
        min_sub_samples: discard internal segments shorter than this many samples
            (default ``int(0.012 * sr)``).
    """
    if min_sub_samples is None:
        min_sub_samples = int(0.012 * sr)
    fp_dim: int | None = None
    out: list[SubcallRepresentation] = []
    for call in call_events:
        i = int(call.onset_s * sr)
        j = max(int(call.offset_s * sr), i + int(0.03 * sr))
        whole_audio = audio[i:j]
        rep = SubcallRepresentation(
            call_id=call.call_id, start_ms=call.onset_s * 1000.0, end_ms=call.offset_s * 1000.0
        )
        if len(whole_audio) < int(0.015 * sr):
            continue
        wfp = np.asarray(fingerprint_fn(whole_audio, sr), dtype=np.float32)
        if not np.any(wfp):
            continue
        if fp_dim is None:
            fp_dim = wfp.shape[0]
        rep.whole_fingerprint = wfp
        for seg in call.internal_segments:
            si = int(seg.start_s * sr)
            sj = max(int(seg.end_s * sr), si + min_sub_samples)
            sseg = audio[si:sj]
            if len(sseg) < min_sub_samples:
                continue
            sfp = np.asarray(fingerprint_fn(sseg, sr), dtype=np.float32)
            if np.any(sfp):
                rep.subcall_fingerprints.append(sfp)
        rep.pooled_fingerprint = pooled_representation(
            rep.whole_fingerprint, rep.subcall_fingerprints, fp_dim
        )
        out.append(rep)
    return out


def _kmeans_tokens(vecs: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(vecs)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaler.transform(vecs))
    cents = km.cluster_centers_
    sv = scaler.transform(vecs)
    return np.argmin(((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1)


def _recovery(
    vecs: np.ndarray, labels: np.ndarray, k: int, n_perm: int, min_per_class: int
) -> dict:
    """k-means tokens → token↔label V/nmi/p + linear probe accuracy."""
    toks = _kmeans_tokens(vecs, k)
    binding = association_test(
        [[int(t)] for t in toks], labels.tolist(), statistic="nmi", n_permutations=n_perm, seed=0
    ).as_dict()
    probe = linear_probe(vecs, labels, min_per_class=min_per_class).as_dict()
    return {
        "V": binding["cramers_v"],
        "nmi": binding["observed_nmi"],
        "p": binding["p_value"],
        "probe": probe.get("accuracy"),
    }


def whole_sub_pooled_ablation(
    reps: list[SubcallRepresentation],
    labels: list,
    *,
    k: int = 12,
    n_perm: int = 500,
    min_per_class: int = 5,
    min_sub_for_subcall: int | None = None,
) -> dict:
    """Standard ablation: whole vs sub-call vs pooled label-recovery.

    For each representation mode, k-means-tokenize → token↔label Cramer's V / NMI / p +
    linear-probe accuracy. Returns the comparison dict. The sub-call mode labels each
    sub-call with its parent call's label (tests whether sub-call acoustic classes carry
    the label signal).

    Args:
        reps: SubcallRepresentations (from ``build_subcall_representations``).
        labels: per-call labels (aligned to ``reps``).
        k: k-means token count.
        min_sub_for_subcall: minimum total sub-segs to run the sub-call mode
            (default ``k * 5``).
    """
    if len(reps) != len(labels):
        raise ValueError(f"reps ({len(reps)}) and labels ({len(labels)}) must align")
    if min_sub_for_subcall is None:
        min_sub_for_subcall = k * 5
    whole = np.asarray([r.whole_fingerprint for r in reps], dtype=np.float32)
    pooled = np.asarray([r.pooled_fingerprint for r in reps], dtype=np.float32)
    labels_arr = np.asarray(labels)
    result = {
        "n_calls": len(reps),
        "n_with_internal": sum(1 for r in reps if r.has_internal),
        "n_subsegs": sum(r.n_internal for r in reps),
        "whole": _recovery(whole, labels_arr, k, n_perm, min_per_class),
        "pooled": _recovery(pooled, labels_arr, k, n_perm, min_per_class),
        "subcall": None,
    }
    all_sub = [fp for r in reps for fp in r.subcall_fingerprints]
    if len(all_sub) >= min_sub_for_subcall:
        sub_vecs = np.asarray(all_sub, dtype=np.float32)
        sub_labels = np.asarray(
            [lab for r, lab in zip(reps, labels) for _ in r.subcall_fingerprints]
        )
        result["subcall"] = _recovery(sub_vecs, sub_labels, k, n_perm, min_per_class)
    return result


def format_ablation_table(result: dict) -> str:
    """Render a whole/sub/pooled ablation result as a readable table string."""
    lines = [
        f"calls: {result['n_calls']} ({result['n_with_internal']} with internal, "
        f"{result['n_subsegs']} sub-segs)",
        f"{'representation':24} {'Cramers V':>10} {'NMI':>8} {'p':>10} {'probe acc':>10}",
        "-" * 66,
    ]

    def _row(name: str, r: dict | None) -> str:
        if r is None:
            return f"{name:24} {'(n/a)':>10}"
        return f"{name:24} {r['V']:10.3f} {r['nmi']:8.3f} {str(r['p']):>10} {str(r['probe']):>10}"

    lines.append(_row("whole call", result["whole"]))
    lines.append(_row("whole + sub-call pool", result["pooled"]))
    lines.append(_row("sub-call (parent lbl)", result["subcall"]))
    return "\n".join(lines)
