import numpy as np
import pytest
from bioacoustic_pipeline.representation.rclat_causal_resample import (
    causal_decimate,
    frame_source_support,
    target_separation,
)


def test_future_perturbation_and_independent_fir_reference():
    x = np.random.default_rng(17).normal(size=6000)
    changed = x.copy()
    changed[3001:] += 100
    a = causal_decimate(x, 48000)
    b = causal_decimate(changed, 48000)
    np.testing.assert_array_equal(a["values"][:1001], b["values"][:1001])
    for k in [32, 33, 100, 500]:
        endpoint = 3 * k
        np.testing.assert_allclose(
            a["values"][k], np.dot(a["taps"], x[endpoint - np.arange(97)]), atol=1e-14, rtol=1e-14
        )
    assert not a["valid"][:32].any() and a["valid"][32:].all()
    np.testing.assert_array_equal(a["latest_source_sample"], np.arange(0, 6000, 3))


def test_support_tracks_filter_history_and_detects_short_horizon_overlap():
    lo, hi = frame_source_support(31)
    assert lo == 3 * (31 * 160) - 96 and hi == 3 * (31 * 160 + 639)
    assert target_separation(31, 4)["disjoint"] is False
    assert target_separation(31, 8)["disjoint"] is True
    assert target_separation(31, 16)["disjoint"] is True
    assert target_separation(31, 8)["gap_source_samples"] == 1826


def test_rejects_unsupported_waveforms_and_indices():
    for x, rate in [
        (np.ones(100), 44100),
        (np.ones((2, 100)), 48000),
        (np.array([np.nan]), 48000),
        (np.array([]), 48000),
    ]:
        with pytest.raises(ValueError):
            causal_decimate(x, rate)
    with pytest.raises(ValueError):
        frame_source_support(-1)
