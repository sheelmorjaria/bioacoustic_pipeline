"""Tests for representation.subcall (whole/sub-call/pooled hierarchical representation).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioacoustic_pipeline.representation.subcall import (
    SubcallRepresentation,
    build_subcall_representations,
    format_ablation_table,
    pooled_representation,
    whole_sub_pooled_ablation,
)
from bioacoustic_pipeline.segmentation.hierarchical import CallEvent, InternalSegment

SR = 8000


def _call(onset_s, offset_s, internal_s=None, call_id="c"):
    internal_s = internal_s or []
    segs = [InternalSegment(parent_call_id=call_id, start_s=s, end_s=e) for s, e in internal_s]
    return CallEvent(call_id=call_id, onset_s=onset_s, offset_s=offset_s, internal_segments=segs)


def test_pooled_representation_dims_with_subcalls():
    whole = np.ones(3, dtype=np.float32)
    subs = [np.zeros(3, dtype=np.float32), 2 * np.ones(3, dtype=np.float32)]
    pooled = pooled_representation(whole, subs, fp_dim=3)
    # whole(3) + mean_sub(3) + [n_sub](1) = 7
    assert pooled.shape == (7,)
    assert pooled[-1] == 2.0  # n_sub marker
    np.testing.assert_allclose(pooled[3:6], np.ones(3))  # mean of [0,0,0] and [2,2,2]


def test_pooled_representation_no_subcalls():
    whole = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pooled = pooled_representation(whole, [], fp_dim=3)
    assert pooled.shape == (7,)
    assert pooled[-1] == 0.0  # n_sub = 0
    np.testing.assert_allclose(pooled[3:6], np.zeros(3))  # zeroed sub-summary
    np.testing.assert_allclose(pooled[:3], whole)  # whole preserved


def test_pooled_representation_infers_fp_dim():
    whole = np.ones(4, dtype=np.float32)
    pooled = pooled_representation(whole, [])
    assert pooled.shape == (4 + 4 + 1,)  # infers fp_dim=4 from whole


def test_subcallrepresentation_flags():
    r0 = SubcallRepresentation(call_id="c", start_ms=0, end_ms=100)
    assert not r0.has_internal and r0.n_internal == 0
    r1 = SubcallRepresentation(
        call_id="c",
        start_ms=0,
        end_ms=100,
        subcall_fingerprints=[np.zeros(3), np.zeros(3)],
    )
    assert r1.has_internal and r1.n_internal == 2


def test_build_subcall_representations_whole_and_sub():
    # audio: loud so fingerprints are nonzero; call 0.0-0.1s with 2 internal segs
    audio = np.ones(int(0.2 * SR), dtype=np.float32)
    fp_fn = lambda seg, sr: np.array([float(seg.mean()), float(seg.std())], dtype=np.float32)  # noqa: E731
    calls = [_call(0.0, 0.1, [(0.03, 0.06), (0.06, 0.09)], call_id="c0")]
    reps = build_subcall_representations(calls, audio, SR, fp_fn)
    assert len(reps) == 1
    r = reps[0]
    assert r.whole_fingerprint.shape == (2,)
    assert len(r.subcall_fingerprints) == 2
    assert r.pooled_fingerprint.shape == (2 + 2 + 1,)
    assert r.pooled_fingerprint[-1] == 2.0
    assert r.has_internal


def test_build_subcall_representations_skips_out_of_bounds():
    audio = np.ones(int(0.2 * SR), dtype=np.float32)  # 0.2 s
    fp_fn = lambda seg, sr: np.array([float(seg.mean())], dtype=np.float32)  # noqa: E731
    # a call whose onset is past the audio end yields an empty slice → skipped
    calls = [_call(1.0, 1.1, call_id="oob"), _call(0.0, 0.1, call_id="ok")]
    reps = build_subcall_representations(calls, audio, SR, fp_fn)
    assert len(reps) == 1
    assert reps[0].call_id == "ok"


def test_build_subcall_representations_skips_silent():
    audio = np.zeros(int(0.2 * SR), dtype=np.float32)  # silent → zero fingerprint → skipped
    fp_fn = lambda seg, sr: np.array([float(seg.mean())], dtype=np.float32)  # noqa: E731
    calls = [_call(0.0, 0.1, call_id="c")]
    reps = build_subcall_representations(calls, audio, SR, fp_fn)
    assert len(reps) == 0  # np.any(fingerprint) is False for silent audio


def test_whole_sub_pooled_ablation_structure():
    # 10 calls, 2 classes (5 each); fingerprints class-separable by amplitude
    audio_a = np.full(int(0.15 * SR), 0.9, dtype=np.float32)
    audio_b = np.full(int(0.15 * SR), 0.1, dtype=np.float32)
    fp_fn = lambda seg, sr: np.array([float(seg.mean())], dtype=np.float32)  # noqa: E731
    reps = []
    for i in range(5):
        reps += build_subcall_representations(
            [_call(0.0, 0.1, [(0.04, 0.07)], call_id=f"a{i}")], audio_a, SR, fp_fn
        )
    for i in range(5):
        reps += build_subcall_representations(
            [_call(0.0, 0.1, [(0.04, 0.07)], call_id=f"b{i}")], audio_b, SR, fp_fn
        )
    labels = ["A"] * 5 + ["B"] * 5
    assert len(reps) == 10
    result = whole_sub_pooled_ablation(reps, labels, k=2, n_perm=10, min_per_class=5)
    assert result["n_calls"] == 10
    assert result["n_with_internal"] == 10
    for mode in ("whole", "pooled"):
        assert set(result[mode]) >= {"V", "nmi", "p", "probe"}
        assert isinstance(result[mode]["V"], float)
    # 10 calls × 1 sub-seg each = 10 sub-segs ≥ k*5=10 → subcall mode runs
    assert result["subcall"] is not None
    assert result["n_subsegs"] == 10


def test_whole_sub_pooled_ablation_misaligned_raises():
    reps = []
    with pytest.raises(ValueError, match="align"):
        whole_sub_pooled_ablation(reps, ["X"], k=2, n_perm=5)


def test_whole_sub_pooled_ablation_subcall_skipped_when_few():
    audio = np.full(int(0.15 * SR), 0.9, dtype=np.float32)
    fp_fn = lambda seg, sr: np.array([float(seg.mean())], dtype=np.float32)  # noqa: E731
    # 6 calls, NO internal segs → 0 sub-segs < k*5 → subcall mode is None
    reps = []
    for i in range(6):
        reps += build_subcall_representations([_call(0.0, 0.1, call_id=f"c{i}")], audio, SR, fp_fn)
    labels = ["A"] * 3 + ["B"] * 3
    result = whole_sub_pooled_ablation(reps, labels, k=2, n_perm=5, min_per_class=3)
    assert result["subcall"] is None
    assert result["n_subsegs"] == 0


def test_format_ablation_table_renders():
    audio_a = np.full(int(0.15 * SR), 0.9, dtype=np.float32)
    audio_b = np.full(int(0.15 * SR), 0.1, dtype=np.float32)
    fp_fn = lambda seg, sr: np.array([float(seg.mean())], dtype=np.float32)  # noqa: E731
    reps, labels = [], []
    for i in range(5):
        reps += build_subcall_representations(
            [_call(0.0, 0.1, [(0.04, 0.07)], call_id=f"a{i}")], audio_a, SR, fp_fn
        )
        labels.append("A")
    for i in range(5):
        reps += build_subcall_representations(
            [_call(0.0, 0.1, [(0.04, 0.07)], call_id=f"b{i}")], audio_b, SR, fp_fn
        )
        labels.append("B")
    result = whole_sub_pooled_ablation(reps, labels, k=2, n_perm=5, min_per_class=5)
    table = format_ablation_table(result)
    assert "whole call" in table
    assert "whole + sub-call pool" in table
    assert "Cramers V" in table
