"""Tests: acoustic-feature trajectory derivatives (gap-safe, unit-correct).

Named features with physical units, per-feature support masks, contiguous
span IDs, first derivatives. Unsupported values remain unavailable (NaN),
never zero-valued acoustic evidence. Caller-mean subtraction before
differentiation is mathematically redundant (exact identity — pinned by
test). No differentiation across invalid gaps. The name->index layout is
pinned against the LIVE extractor (regression for the v1 drift).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_UNITS,
    first_derivative,
    supported_features,
    trajectory_derivatives,
)


class TestFeatureSchema(unittest.TestCase):
    def test_supported_features_have_units(self):
        feats = supported_features()
        self.assertGreater(len(feats), 5)
        for name in feats:
            self.assertIn(name, FEATURE_UNITS)
            self.assertTrue(FEATURE_UNITS[name])  # non-empty unit string

    def test_duration_not_in_supported(self):
        # duration is constant for equal-length windows; derivative is
        # uninformative — excluded by the schema
        self.assertNotIn("duration_ms", supported_features())

    def test_schema_version_stamped(self):
        self.assertEqual(FEATURE_SCHEMA_VERSION, 2)

    def test_layout_pinned_to_extractor(self):
        """The name->index map must match the LIVE extractor layout.

        Regression: v1 assumed a stale layout whose env_cv@18 actually read
        macro's mod_depth (constant 0 on 40 ms windows). Anchors below pin
        block boundaries and the drifted slot.
        """
        from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
        from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
            _FEATURE_ORDER,
            _NAME_TO_IDX,
        )

        sr = 250000
        t = np.arange(int(0.04 * sr)) / sr
        am = (0.5 + 0.5 * np.sin(2 * np.pi * 80 * t)) * np.sin(2 * np.pi * 40000 * t)
        c = afp.extract(am, sr, species="bat").combined
        self.assertEqual(len(c), 44)
        self.assertEqual(len(_FEATURE_ORDER), 44)
        # physics anchors
        self.assertAlmostEqual(c[_NAME_TO_IDX["dur_ms"]], 40.0, places=3)
        self.assertAlmostEqual(c[_NAME_TO_IDX["dom_freq"]], 40000.0, places=1)
        self.assertAlmostEqual(c[_NAME_TO_IDX["centroid"]], 40000.0, places=0)
        # the previously-drifted slot: envelope CV is live on an AM tone,
        # while macro's mod_depth sits at 18
        self.assertGreater(c[_NAME_TO_IDX["dyn_cv"]], 0.0)
        self.assertEqual(_FEATURE_ORDER[18], "mod_depth")
        self.assertEqual(_FEATURE_ORDER[38], "dyn_cv")
        # macro/micro block boundaries
        self.assertEqual(_FEATURE_ORDER[15], "curvature")
        self.assertEqual(_FEATURE_ORDER[29], "d_mean")


class TestFirstDerivative(unittest.TestCase):
    def test_constant_trajectory_zero_derivative(self):
        x = np.full(10, 5.0)
        valid = np.ones(10, dtype=bool)
        d = first_derivative(x, valid, dt=0.01)
        self.assertTrue(np.allclose(d[1:], 0.0))

    def test_linear_trajectory(self):
        x = np.arange(10, dtype=np.float64) * 3.0  # slope 3.0/unit-time
        valid = np.ones(10, dtype=bool)
        d = first_derivative(x, valid, dt=0.01)
        self.assertTrue(np.allclose(d[1:], 3.0 / 0.01))

    def test_seconds_vs_milliseconds(self):
        x = np.arange(10, dtype=np.float64) * 2.0
        valid = np.ones(10, dtype=bool)
        d_s = first_derivative(x, valid, dt=1.0)  # dt = 1s
        d_ms = first_derivative(x, valid, dt=0.001)  # dt = 1ms
        # 1ms spacing gives 1000x larger derivative
        ratio = d_ms[5] / d_s[5]
        self.assertAlmostEqual(ratio, 1000.0, places=6)

    def test_no_derivative_across_invalid_gap(self):
        x = np.arange(10, dtype=np.float64)
        valid = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        d = first_derivative(x, valid, dt=0.01)
        # the derivative crossing the gap must be NaN
        self.assertTrue(np.isnan(d[6]))
        # derivatives within valid runs are correct
        self.assertAlmostEqual(d[1], 1.0 / 0.01)
        self.assertAlmostEqual(d[7], 1.0 / 0.01)

    def test_invalid_positions_are_nan(self):
        x = np.arange(10, dtype=np.float64)
        valid = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
        d = first_derivative(x, valid, dt=0.01)
        self.assertTrue(np.isnan(d[2]))
        self.assertTrue(np.isnan(d[0]))  # first position always NaN (no backward)

    def test_caller_mean_redundancy_identity(self):
        """Subtracting a constant caller mean before differentiation is
        mathematically redundant — the derivatives are identical."""
        x = np.random.RandomState(0).randn(20) * 10
        valid = np.ones(20, dtype=bool)
        mu = 42.7  # arbitrary caller mean
        d_raw = first_derivative(x, valid, dt=0.01)
        d_centered = first_derivative(x - mu, valid, dt=0.01)
        self.assertTrue(np.allclose(d_raw, d_centered, equal_nan=True))

    def test_sin_derivative(self):
        t = np.arange(0, 1, 0.01)
        x = np.sin(2 * np.pi * 5 * t)
        valid = np.ones(len(t), dtype=bool)
        d = first_derivative(x, valid, dt=0.01)
        midpoints = (t[1:] + t[:-1]) / 2
        expected_mid = 2 * np.pi * 5 * np.cos(2 * np.pi * 5 * midpoints)
        self.assertTrue(np.allclose(d[1:], expected_mid, atol=0.5))


class TestTrajectoryDerivatives(unittest.TestCase):
    def test_full_pipeline_with_masks(self):
        # 5 frames of a trajectory with one invalid frame
        traj = np.random.RandomState(1).randn(5, 44)
        valid = np.array([1, 1, 0, 1, 1], dtype=bool)
        result = trajectory_derivatives(traj, valid, dt=0.01)
        self.assertIn("derivatives", result)
        self.assertIn("valid", result)
        self.assertIn("span_ids", result)
        self.assertIn("units", result)
        self.assertEqual(result["feature_schema_version"], FEATURE_SCHEMA_VERSION)
        # derivative at invalid position is NaN
        self.assertTrue(np.isnan(result["derivatives"][2, 0]))
        # two valid spans
        self.assertEqual(result["span_ids"][0], 0)
        self.assertEqual(result["span_ids"][3], 1)
        self.assertEqual(result["span_ids"][2], -1)  # invalid

    def test_padding_inert(self):
        traj = np.random.RandomState(2).randn(5, 44)
        valid = np.ones(5, dtype=bool)
        a = trajectory_derivatives(traj, valid, dt=0.01)
        padded_traj = np.vstack([traj, np.zeros((3, 44))])
        padded_valid = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        b = trajectory_derivatives(padded_traj, padded_valid, dt=0.01)
        # derivatives on valid positions unchanged
        np.testing.assert_allclose(a["derivatives"], b["derivatives"][:5], equal_nan=True)

    def test_unsupported_features_excluded(self):
        traj = np.random.RandomState(3).randn(5, 6)
        valid = np.ones(5, dtype=bool)
        result = trajectory_derivatives(
            traj, valid, dt=0.01, feature_names=["centroid_hz", "rms", "a", "b", "c", "d"]
        )
        # only supported features get derivatives
        self.assertLess(result["derivatives"].shape[1], 6)


if __name__ == "__main__":
    unittest.main()
