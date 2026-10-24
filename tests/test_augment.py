"""Tests: training-time audio augmentation (embeddings.augment).

Mechanics only: measured SNR hits target, determinism, monotonicity, dtype
preservation, zero-signal guard. Numpy only.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.embeddings.augment import add_white_noise, random_gain


def _signal(seconds=1.0, sr=8000, freq=440.0):
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestAddWhiteNoise(unittest.TestCase):
    def test_measured_snr_hits_target(self):
        for snr_db in (20.0, 10.0, 0.0):
            with self.subTest(snr_db=snr_db):
                x = _signal()
                out = add_white_noise(x, snr_db=snr_db, rng=np.random.RandomState(0))
                noise = out.astype(np.float64) - x.astype(np.float64)
                p_sig = float((x.astype(np.float64) ** 2).mean())
                p_noise = float((noise**2).mean())
                measured = 10.0 * np.log10(p_sig / p_noise)
                self.assertAlmostEqual(measured, snr_db, delta=0.2)

    def test_deterministic_under_seed(self):
        x = _signal()
        a = add_white_noise(x, snr_db=10.0, rng=np.random.RandomState(7))
        b = add_white_noise(x, snr_db=10.0, rng=np.random.RandomState(7))
        np.testing.assert_array_equal(a, b)

    def test_shape_and_dtype_preserved(self):
        x = _signal()
        out = add_white_noise(x, snr_db=10.0, rng=np.random.RandomState(0))
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_lower_snr_means_more_noise_energy(self):
        x = _signal()
        energies = []
        for snr in (30.0, 10.0, -5.0):
            out = add_white_noise(x, snr_db=snr, rng=np.random.RandomState(1))
            noise = out.astype(np.float64) - x.astype(np.float64)
            energies.append(float((noise**2).mean()))
        self.assertLess(energies[0], energies[1])
        self.assertLess(energies[1], energies[2])

    def test_zero_signal_guard_returns_input(self):
        x = np.zeros(1000, dtype=np.float32)
        out = add_white_noise(x, snr_db=10.0, rng=np.random.RandomState(0))
        np.testing.assert_array_equal(out, x)


class TestRandomGain(unittest.TestCase):
    def test_gain_within_bounds_and_deterministic(self):
        x = _signal()
        a = random_gain(x, lo=0.5, hi=1.5, rng=np.random.RandomState(3))
        b = random_gain(x, lo=0.5, hi=1.5, rng=np.random.RandomState(3))
        np.testing.assert_array_equal(a, b)
        ratio = a.astype(np.float64) / x.astype(np.float64)
        ratio = ratio[np.isfinite(ratio)]  # sine zero-crossings give 0/0
        self.assertGreaterEqual(float(ratio.min()), 0.5 - 1e-6)
        self.assertLessEqual(float(ratio.max()), 1.5 + 1e-6)

    def test_invalid_bounds_raise(self):
        x = _signal()
        with self.assertRaises(ValueError):
            random_gain(x, lo=1.5, hi=0.5, rng=np.random.RandomState(0))


if __name__ == "__main__":
    unittest.main()
