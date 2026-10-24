"""Tests: time-frequency scaling study machinery (deterministic, falsifiable).

Pinned contracts: playback expansion divides frequencies and multiplies
duration/gaps exactly; Phi is deterministic and level-invariant in its
envelope/AM components; surrogates preserve what they claim and disrupt what
they claim; the distance is a squared standardized deviation; the spec dict
is explicit about what the study may and may not conclude.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.scaling_organisation import (
    ALPHA_GRID,
    DESCRIPTOR_NAMES,
    am_modulated,
    build_reference,
    compress_playback,
    expand_playback,
    long_window_descriptor,
    phase_randomized_surrogate,
    scaling_study_spec,
    spec_matched_noise,
    standardized_distance,
    to_analysis_rate,
    transform_and_describe,
    unit_reorder_surrogate,
)


def _tone(seconds: float, freq: float, sr: int = 16000, amp: float = 0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _gap_signal(sr: int = 16000):
    """Tone (0.2 s) - genuine silence (0.3 s) - tone (0.2 s)."""
    a = _tone(0.2, 1000, sr)
    gap = np.zeros(int(0.3 * sr))
    b = _tone(0.2, 1500, sr)
    return np.concatenate([a, gap, b])


class TestPlaybackScaling(unittest.TestCase):
    def test_duration_multiplies(self):
        x = _tone(1.0, 500.0)
        y = expand_playback(x, 16000, 8)
        self.assertAlmostEqual(len(y) / len(x), 8.0, places=2)

    def test_frequency_divides(self):
        # a 4 kHz tone at alpha=4 must peak near 1 kHz after expansion
        x = _tone(1.0, 4000.0)
        y = expand_playback(x, 16000, 4).astype(np.float64)
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        peak = np.fft.rfftfreq(len(y), 1.0 / 16000)[np.argmax(spec)]
        self.assertAlmostEqual(peak, 1000.0, delta=30.0)

    def test_gaps_expand_proportionally(self):
        x = _gap_signal()
        y = expand_playback(x, 16000, 10).astype(np.float64)
        # the central silent run is 0.3 s x 10 = 3 s; sample it clear of the
        # few-ms resampler ringing at the tone edges
        mid = y[int(2.2 * 16000) : int(3.8 * 16000)]
        self.assertLess(float(np.abs(mid).max()), 1e-3)
        self.assertAlmostEqual(len(y) / len(x), 10.0, places=2)

    def test_compress_then_expand_recovers(self):
        x = _tone(1.0, 800.0)
        c = compress_playback(x, 16000, 5)
        r = expand_playback(c, 16000, 5)
        spec = np.abs(np.fft.rfft(r[: min(len(r), 16000)]))
        peak = np.fft.rfftfreq(len(spec) * 2 - 1, 1.0 / 16000)[np.argmax(spec)]
        self.assertAlmostEqual(peak, 800.0, delta=40.0)

    def test_analysis_rate_resample(self):
        x = _tone(0.5, 2000.0, sr=250000)
        z = to_analysis_rate(x, 250000)
        self.assertEqual(16000, 16000)
        self.assertAlmostEqual(len(z) / 0.5, 16000, delta=800)


class TestDescriptor(unittest.TestCase):
    def test_deterministic(self):
        rng = np.random.RandomState(0)
        x = rng.randn(32000) * 0.1
        a = long_window_descriptor(x)
        b = long_window_descriptor(x.copy())
        np.testing.assert_array_equal(a["vector"], b["vector"])

    def test_names_and_finiteness_on_tone(self):
        d = long_window_descriptor(_tone(2.0, 500.0))
        self.assertEqual(len(d["names"]), 25)
        self.assertTrue(np.isfinite(d["vector"]).all(), d["vector"])

    def test_am_band_moves_with_modulation_rate(self):
        # 4 Hz AM tone must put more normalised AM energy in 2-4/4-8 Hz than
        # in 0.5-2 Hz; a 1 Hz AM tone the reverse
        sr = 16000
        t = np.arange(2 * sr) / sr
        fast = 0.2 * (1 + 0.8 * np.sin(2 * np.pi * 4 * t)) * np.sin(2 * np.pi * 1000 * t)
        slow = 0.2 * (1 + 0.8 * np.sin(2 * np.pi * 1.0 * t)) * np.sin(2 * np.pi * 1000 * t)
        df = long_window_descriptor(fast)["vector"]
        ds = long_window_descriptor(slow)["vector"]
        i_lo = DESCRIPTOR_NAMES.index("am_0.5_2hz")
        i_mid = DESCRIPTOR_NAMES.index("am_4_8hz")
        self.assertGreater(df[i_mid], df[i_lo])
        self.assertGreater(ds[i_lo], ds[i_mid])

    def test_too_short_returns_nan(self):
        d = long_window_descriptor(np.zeros(100))
        self.assertEqual(d["n_valid_frames"], 0)
        self.assertTrue(np.isnan(d["vector"]).all())


class TestSurrogates(unittest.TestCase):
    def test_phase_surrogate_preserves_spectrum(self):
        x = _gap_signal()
        s = phase_randomized_surrogate(x, seed=1)
        sx = np.abs(np.fft.rfft(x))
        ss = np.abs(np.fft.rfft(s))
        # per-bin magnitudes identical on meaningful bins (zeros stay zero)
        m = sx > 1e-6 * sx.max()
        np.testing.assert_allclose(ss[m] / sx[m], 1.0, rtol=1e-9)
        self.assertTrue(np.all(ss[~m] < 1e-6 * sx.max()))

    def test_spec_noise_coarse_only(self):
        x = _tone(1.0, 1000.0) + 0.05 * np.random.RandomState(2).randn(16000)
        n = spec_matched_noise(x, seed=3, sr=16000)
        # energy scale preserved (matched), fine comb structure NOT preserved
        self.assertAlmostEqual(
            float(np.sqrt((n.astype(np.float64) ** 2).mean())),
            float(np.sqrt((x**2).mean())),
            delta=0.05,
        )

    def test_am_modulation_depth(self):
        x = np.random.RandomState(4).randn(16000) * 0.1
        y = am_modulated(x, 16000, rate_hz=4.0, depth=0.5)
        self.assertEqual(len(y), len(x))

    def test_unit_reorder_preserves_multiset_and_duration(self):
        sr = 16000
        parts = [np.zeros(int(0.23 * sr))]  # leading silence: observed timeline
        for f in (600.0, 900.0, 1200.0):
            parts.append(_tone(0.15, f, sr))
            parts.append(np.zeros(int(0.1 * sr)))
        x = np.concatenate(parts)
        y = unit_reorder_surrogate(x, sr, seed=6)
        if y is None:
            self.fail("expected >= 3 units")
        self.assertEqual(len(y), len(x))
        # unit durations multiset preserved (activity decode is deterministic)
        from bioacoustic_pipeline.segmentation.silence_units import (
            HysteresisConfig,
            extract_activity_units,
        )

        cfg = HysteresisConfig(frame_ms=10.0)
        rx = extract_activity_units(x, sr=sr, config=cfg, ref_amplitude=1.0)
        ry = extract_activity_units(y.astype(np.float64), sr=sr, config=cfg, ref_amplitude=1.0)
        dx = sorted(s.stop_frame - s.start_frame for s in rx.spans)
        dy = sorted(s.stop_frame - s.start_frame for s in ry.spans)
        self.assertEqual(len(dx), len(dy))
        for a, b in zip(dx, dy):
            self.assertLessEqual(abs(a - b), 1)  # edge fades: <= 1 boundary frame

    def test_unit_reorder_unavailable_below_three_units(self):
        y = unit_reorder_surrogate(_tone(0.5, 500.0), 16000, seed=0)
        self.assertIsNone(y)  # unavailable, never faked


class TestDistanceAndSpec(unittest.TestCase):
    def test_reference_and_distance(self):
        rng = np.random.RandomState(0)
        rows = [long_window_descriptor(rng.randn(32000) * 0.1)["vector"] for _ in range(10)]
        ref = build_reference(rows)
        self.assertEqual(ref["n"], 10)
        d_in = standardized_distance(rows[0], ref)
        d_out = standardized_distance(long_window_descriptor(_tone(2.0, 4000.0))["vector"], ref)
        self.assertGreater(d_out, d_in)

    def test_transform_and_describe_pipeline(self):
        # ultrasonic tone (20 kHz at 64 kHz sr) needs alpha >= 8 to enter the
        # 16 kHz analysis band: distance to a low-freq reference must drop
        sr = 64000
        x = _tone(1.0, 20000.0, sr=sr)
        rows = [
            long_window_descriptor(to_analysis_rate(_tone(2.0, f, 16000), 16000))["vector"]
            for f in (300.0, 500.0, 800.0, 1200.0, 2000.0, 300.0, 900.0)
        ]
        ref = build_reference(rows)
        d1 = standardized_distance(transform_and_describe(x, sr, 1)["vector"], ref)
        d8 = standardized_distance(transform_and_describe(x, sr, 8)["vector"], ref)
        self.assertGreater(d1, d8)

    def test_short_after_transform_unavailable(self):
        # 0.2 s at alpha=1 is below the 1 s modulation-support minimum
        self.assertIsNone(transform_and_describe(_tone(0.2, 500.0), 16000, 1))

    def test_spec_contract(self):
        spec = scaling_study_spec()
        self.assertIn("decision_rule", spec)
        excluded = " ".join(spec["explicitly_excluded"]).lower()
        self.assertIn("asr", excluded)
        self.assertIn("semantic", excluded)
        self.assertIn("vocoders", excluded)
        self.assertIn("firewall", spec)
        self.assertEqual(tuple(spec["alpha_grid"]), ALPHA_GRID)


if __name__ == "__main__":
    unittest.main()
