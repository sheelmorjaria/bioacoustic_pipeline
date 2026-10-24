import numpy as np
from bioacoustic_pipeline.diagnostics.rclat_filter_response import response_metrics


def test_unity_filter_and_reference_complex_transform():
    r = response_metrics(np.array([1.0]), points=1025)
    assert r["passband_max_abs_dB"] == 0 and r["stopband_attenuation_dB"] == 0
    taps = np.array([0.25, 0.5, 0.25])
    r = response_metrics(taps, points=1025)
    for i in [0, 100, 600, 1024]:
        f = r["frequency_hz"][i]
        reference = sum(taps[k] * np.exp(-2j * np.pi * f * k / 48000) for k in range(3))
        np.testing.assert_allclose(r["complex_response"][i], reference, atol=1e-14)


def test_bad_alias_filter_fails_fixed_minimum_suppression():
    r = response_metrics(np.array([1.0]), points=1025)
    assert not r["minimum_engineering_requirements_passed"]
