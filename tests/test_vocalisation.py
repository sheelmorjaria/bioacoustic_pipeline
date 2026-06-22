"""Tests for representation.vocalisation (hierarchical dual-stream token).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioacoustic_pipeline.representation.vocalisation import (
    DualStreamSubToken,
    DualStreamVocalisation,
    build_vocalisation,
    format_4way_table,
    whole_sub_pooled_sequence_ablation,
)
from bioacoustic_pipeline.segmentation.hierarchical import CallEvent, InternalSegment

SR = 8000


def _call(onset_s, offset_s, internal_s=None, call_id="c"):
    segs = [
        InternalSegment(parent_call_id=call_id, start_s=s, end_s=e) for s, e in (internal_s or [])
    ]
    return CallEvent(
        call_id=call_id,
        onset_s=onset_s,
        offset_s=offset_s,
        crnn_confidence=0.9,
        internal_segments=segs,
    )


def test_subtoken_duration_and_relatives():
    sub = DualStreamSubToken(
        parent_vocalisation_id="v",
        start_s=0.1,
        end_s=0.2,
        relative_start=0.1,
        relative_end=0.2,
        boundary_type="internal",
    )
    assert sub.duration_s == pytest.approx(0.1)
    assert sub.as_dict()["boundary_type"] == "internal"


def test_vocalisation_properties_and_structural_summary():
    voc = DualStreamVocalisation(
        vocalisation_id="v",
        start_s=1.0,
        end_s=2.0,
        sub_segments=[
            DualStreamSubToken(relative_end=0.4),
            DualStreamSubToken(relative_end=0.7),
        ],
    )
    assert voc.duration_s == pytest.approx(1.0)
    assert voc.subcall_count == 2
    # relative boundary positions = the internal cuts between segments (exclude edges 0/1)
    assert voc.relative_boundary_positions == [0.4, 0.7]
    summ = voc.structural_summary()
    assert summ["subcall_count"] == 2
    assert summ["relative_boundary_positions"] == [0.4, 0.7]
    # mean internal gap over [0, 0.4, 0.7, 1.0] → gaps 0.4,0.3,0.3 → mean 0.333
    assert summ["mean_internal_gap"] == pytest.approx(1 / 3, abs=0.01)
    assert "duration_s" in summ  # species-agnostic structural summary (no token IDs)


def test_vocalisation_no_internal_has_empty_boundaries():
    voc = DualStreamVocalisation(vocalisation_id="v", start_s=0.0, end_s=0.1)
    assert voc.subcall_count == 0
    assert voc.relative_boundary_positions == []
    assert voc.structural_summary()["mean_internal_gap"] == 1.0


def test_build_vocalisation_threads_subtokens():
    audio = np.ones(int(0.5 * SR), dtype=np.float32)
    fp_fn = lambda seg, sr: np.array([float(seg.mean()), float(seg.std())], dtype=np.float32)  # noqa: E731
    # call 0.0-0.4s with 2 internal segs at relative ~0.25, 0.5
    call = _call(0.0, 0.4, [(0.1, 0.2), (0.2, 0.3)], call_id="c0")
    voc = build_vocalisation(
        call,
        audio,
        SR,
        fingerprint_fn=fp_fn,
        species="marmoset",
        calltype="Phee",
        caller_id="twin3A",
    )
    assert voc is not None
    assert voc.species == "marmoset" and voc.calltype == "Phee" and voc.caller_id == "twin3A"
    assert voc.acoustic_fingerprint.shape == (2,)
    assert voc.subcall_count == 2
    for sub in voc.sub_segments:
        assert sub.parent_vocalisation_id == "c0"
        assert 0.0 <= sub.relative_start <= sub.relative_end <= 1.0
        assert sub.acoustic_fingerprint.size > 0


def test_build_vocalisation_skips_silent():
    # silent audio → zero whole fingerprint → build returns None
    audio = np.zeros(int(0.2 * SR), dtype=np.float32)
    fp_fn = lambda seg, sr: np.array([float(seg.mean())], dtype=np.float32)  # noqa: E731
    voc = build_vocalisation(_call(0.0, 0.1), audio, SR, fingerprint_fn=fp_fn)
    assert voc is None


def test_4way_ablation_runs():
    audio = np.full(int(0.2 * SR), 0.9, dtype=np.float32)
    fp_fn = lambda seg, sr: np.array([float(seg.mean()), float(seg.std())], dtype=np.float32)  # noqa: E731
    vocs, labels = [], []
    for i in range(6):
        vocs.append(
            build_vocalisation(
                _call(0.0, 0.15, [(0.05, 0.1)], call_id=f"a{i}"), audio, SR, fingerprint_fn=fp_fn
            )
        )
        labels.append("A")
    for i in range(6):
        vocs.append(
            build_vocalisation(
                _call(0.0, 0.15, [(0.05, 0.1)], call_id=f"b{i}"),
                np.full(int(0.2 * SR), 0.1, dtype=np.float32),
                SR,
                fingerprint_fn=fp_fn,
            )
        )
        labels.append("B")
    result = whole_sub_pooled_sequence_ablation(vocs, labels, k=2)
    assert result["n_vocs"] == 12
    assert set(result) >= {"whole", "pooled", "sequence", "subcall"}
    # sequence vector is constant-dim across calls (whole + k histogram + 2)
    for mode in ("whole", "pooled", "sequence"):
        assert set(result[mode]) >= {"V", "nmi", "p", "probe"}
    table = format_4way_table(result)
    assert "whole + sub SEQUENCE" in table


def test_4way_ablation_misaligned_raises():
    with pytest.raises(ValueError, match="align"):
        whole_sub_pooled_sequence_ablation([], ["X"], k=2)


if __name__ == "__main__":
    import unittest

    unittest.main()
