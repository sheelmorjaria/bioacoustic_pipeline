import numpy as np
from bioacoustic_pipeline.representation.rclat_causal_resample_v2 import (
    causal_decimate_v2,
    mel_features_v2,
    target_separation_v2,
)
from bioacoustic_pipeline.representation.rclat_ctc import acoustic_features


def test_independent_fir_prefix_and_startup():
    x = np.random.default_rng(19).normal(size=48000)
    a = causal_decimate_v2(x, 48000)
    changed = x.copy()
    changed[30001:] += 10
    b = causal_decimate_v2(changed, 48000)
    np.testing.assert_array_equal(a["values"][:10001], b["values"][:10001])
    assert not a["valid"][:64].any() and a["valid"][64:].all()
    for k in [64, 100, 500]:
        np.testing.assert_allclose(
            a["values"][k], np.dot(a["taps"], x[3 * k - np.arange(193)]), atol=1e-14
        )
    assert target_separation_v2(32, 8)["gap_source_samples"] == 1730
    assert target_separation_v2(32, 16)["gap_source_samples"] == 5570
    assert not target_separation_v2(32, 4)["disjoint"]


def test_frontend_values_are_original_mels_with_explicit_startup_mask():
    x = np.random.default_rng(20).normal(size=48000)
    a = causal_decimate_v2(x, 48000)
    f = mel_features_v2(x, 48000)
    np.testing.assert_array_equal(f["mel"], acoustic_features(a["values"], 16000)["mel"])
    assert f["mel"].shape == (97, 64) and not f["mel_valid"][0].any() and f["mel_valid"][1:].all()
    assert f["source_first_samples"][0] == -192 and f["source_last_samples"][0] == 1917
    changed = x.copy()
    changed[30001:] += 10
    g = mel_features_v2(changed, 48000)
    rows = f["source_last_samples"] <= 30000
    np.testing.assert_array_equal(f["mel"][rows], g["mel"][rows])
