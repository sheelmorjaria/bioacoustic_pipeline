"""Unit tests for the alternative-structure modules (RQA, PST). Pure-stdlib/numpy — no audio."""

import numpy as np

from bioacoustic_pipeline.diagnostics.structural_alternatives import (
    probabilistic_suffix_tree as pst,
)
from bioacoustic_pipeline.diagnostics.structural_alternatives import rqa


def _periodic(n=120, d=8):
    """A trajectory that loops through a fixed cycle of states → high determinism."""
    rng = np.random.RandomState(0)
    centers = rng.randn(6, d)
    idx = np.arange(n) % 6
    return centers[idx] + rng.randn(n, d) * 0.01


def _random(n=120, d=8):
    rng = np.random.RandomState(1)
    return rng.randn(n, d)


def test_rqa_periodic_more_deterministic_than_random():
    rp_det = rqa.rqa_for_trajectory(_periodic(), target_rec=0.05)["determinism"]
    rp_rand = rqa.rqa_for_trajectory(_random(), target_rec=0.05)["determinism"]
    assert rp_det > rp_rand, (rp_det, rp_rand)
    assert rp_det > 0.3  # a cycling signal is substantially deterministic


def test_recurrence_plot_density_near_target():
    rp, _ = rqa.recurrence_plot(_random(100, 6), target_rec=0.05)
    n = rp.shape[0]
    density = rp.sum() / (n * n)
    assert 0.02 < density < 0.12, density


def test_rqa_short_input_safe():
    out = rqa.rqa_for_trajectory(np.zeros((1, 4)))
    assert out["determinism"] == 0.0
    assert out["n_frames"] == 1


def test_pst_finds_structure_in_cyclic_sequence():
    seq = [[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]]
    out = pst.fit_pst(seq, max_depth=3, pmin=0.2, alpha=0.3, min_count=2)
    assert out["n_significant_nodes"] >= 1
    assert out["max_depth_reached"] >= 1


def test_pst_empty_safe():
    out = pst.fit_pst([], max_depth=3)
    assert out["n_significant_nodes"] == 0
    assert out["entropy_rate"] is None
