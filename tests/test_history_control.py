"""Tests: suffix-preserving history controls (§46 bat temporal track).

The control sequence [H'_early, H_suffix, Target] keeps the immediate suffix
(covering the AR receptive field) and the exact target byte-identical, and
replaces only the early history — via time-reversal, a matched donor, or
context shuffle. Splice continuity is enforced with a short crossfade so no
artificial discontinuity is introduced at the junction (a naive splice would
hand the AR a fabricated "event" and fake a history effect).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.embeddings.history_control import (
    build_control_history,
    crossfade_splice,
)


def _audio(n=600, sr=8000, f0=300.0, seed=0):
    t = np.arange(n) / sr
    sig = 0.5 * np.sin(2 * np.pi * f0 * t)
    rng = np.random.RandomState(seed)
    return (sig + 0.01 * rng.randn(n)).astype(np.float32)


class TestCrossfadeSplice(unittest.TestCase):
    def test_suffix_bytes_identical_beyond_blend(self):
        early, suffix = _audio(200), _audio(300, f0=500.0)
        out = crossfade_splice(early, suffix, blend=12)
        self.assertEqual(len(out), len(early) + len(suffix))
        np.testing.assert_array_equal(out[len(early) + 12 :], suffix[12:])

    def test_junction_step_vs_deliberately_discontinuous_splice(self):
        # force a naive junction step: early ends at +0.9, suffix starts at -0.9
        early = np.concatenate([_audio(188), np.full(12, 0.9, dtype=np.float32)])
        suffix = np.concatenate([np.full(12, -0.9, dtype=np.float32), _audio(288, seed=2)])
        i0 = len(early)
        naive_step = abs(
            float(np.concatenate([early, suffix])[i0 - 1])
            - float(np.concatenate([early, suffix])[i0])
        )
        blended = crossfade_splice(early, suffix, blend=12)
        junction_step = abs(float(blended[i0 - 1]) - float(blended[i0]))
        self.assertLess(junction_step, naive_step / 4)  # fade kills the junction step

    def test_blend_is_convex_combination(self):
        early, suffix = _audio(200, f0=200.0), _audio(300, f0=900.0, seed=1)
        out = crossfade_splice(early, suffix, blend=16)
        i0, blend = len(early), 16
        lo = np.minimum(early[-blend:], suffix[:blend])
        hi = np.maximum(early[-blend:], suffix[:blend])
        np.testing.assert_array_less(out[i0 - blend : i0] - 1e-6, hi)
        np.testing.assert_array_less(lo - 1e-6, out[i0 - blend : i0] + 1e-6)


class TestBuildControlHistory(unittest.TestCase):
    def _parts(self):
        seq = _audio(600, f0=300.0)
        n_early, n_suffix = 300, 300
        return seq, n_early, n_suffix

    def test_suffix_and_target_identical(self):
        seq, n_early, n_suffix = self._parts()
        donor = _audio(300, f0=350.0, seed=1)
        ctrl = build_control_history(seq, n_early=n_early, mode="donor", donor=donor, blend=12)
        self.assertEqual(len(ctrl), len(seq))
        np.testing.assert_array_equal(ctrl[n_early:], seq[n_early:])  # suffix kept

    def test_reverse_mode(self):
        seq, n_early, n_suffix = self._parts()
        ctrl = build_control_history(seq, n_early=n_early, mode="reverse", blend=0)
        np.testing.assert_array_equal(ctrl[:n_early], seq[:n_early][::-1])

    def test_shuffle_mode_reproducible(self):
        seq, n_early, n_suffix = self._parts()
        a = build_control_history(seq, n_early=n_early, mode="shuffle", seed=3, blend=0)
        b = build_control_history(seq, n_early=n_early, mode="shuffle", seed=3, blend=0)
        np.testing.assert_array_equal(a[:n_early], b[:n_early])
        self.assertFalse(np.array_equal(a[:n_early], seq[:n_early]))

    def test_donor_required_for_donor_mode(self):
        seq, n_early, _ = self._parts()
        with self.assertRaises(ValueError):
            build_control_history(seq, n_early=n_early, mode="donor")

    def test_no_short_histories(self):
        seq, n_early, _ = self._parts()
        with self.assertRaises(ValueError):
            build_control_history(seq[:5], n_early=5, mode="reverse")


if __name__ == "__main__":
    unittest.main()
