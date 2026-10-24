"""Tests: spectral annotation proposals (different family from v1).

The proposals are generated from time-frequency excess energy — a different
detection family than the frozen v1 RMS/hysteresis detector. Tests verify:
clean tone recovery, silence rejection, background suppression, run merging,
and that the method is deterministic.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.ingestion.annotation_proposals import (
    activity_runs,
    spectral_proposals,
    spectrogram_excess,
)


def _signal(sr=250000, dur=0.5, f0=30000):
    t = np.arange(int(sr * dur)) / sr
    return 0.5 * np.sin(2 * np.pi * f0 * t)


def _bursts(sr=250000):
    """Silence-tone-silence-gap-tone-silence (two disjoint events)."""
    tone = _signal(sr=sr, dur=0.1, f0=30000)
    silence = np.zeros(int(sr * 0.15))
    gap = np.zeros(int(sr * 0.08))
    return np.concatenate([silence, tone, gap, tone, silence])


class TestSpectralExcess(unittest.TestCase):
    def test_tone_in_silence_produces_proposal(self):
        # a 100%-duty tone becomes background under median-over-time; the
        # actual use case is tone embedded in silence
        sr = 250000
        x = np.concatenate(
            [np.zeros(int(sr * 0.1)), _signal(sr=sr, dur=0.1), np.zeros(int(sr * 0.1))]
        )
        p = spectral_proposals(x, sr)
        self.assertGreaterEqual(len(p), 1)
        self.assertGreater(p[0]["peak_z"], 4.0)

    def test_silence_produces_no_proposals(self):
        p = spectral_proposals(np.zeros(125000), 250000)
        self.assertEqual(len(p), 0)

    def test_two_bursts_two_proposals(self):
        p = spectral_proposals(_bursts(), 250000)
        self.assertEqual(len(p), 2)

    def test_background_suppression(self):
        """Stationary noise floor is suppressed; tone above it is detected."""
        sr = 250000
        t = np.arange(int(sr * 0.3)) / sr
        noise = 0.01 * np.random.RandomState(0).randn(len(t))
        tone = np.zeros_like(noise)
        tone[: int(sr * 0.1)] += 0.3 * np.sin(2 * np.pi * 25e3 * t[: int(sr * 0.1)])
        p = spectral_proposals(tone + noise, sr)
        self.assertGreaterEqual(len(p), 1)
        self.assertGreater(p[0]["peak_z"], 3.0)

    def test_deterministic(self):
        x = _bursts()
        a = spectral_proposals(x, 250000)
        b = spectral_proposals(x, 250000)
        self.assertEqual(a, b)

    def test_different_from_v1_rms_family(self):
        """The proposal method's temporal grid is spectrogram hops, not the
        v1 detector's 10 ms RMS frames — different by construction."""
        Z, freqs, times = spectrogram_excess(_signal(), 250000)
        # the spectrogram temporal grid has ~250000/128 = ~1953 columns
        # vs the v1 RMS grid's 250000/2500 = 100 frames — different grids
        self.assertGreater(len(times), 10 * (len(_signal()) // 2500))


class TestRunMerging(unittest.TestCase):
    def test_short_gap_merged(self):
        # 5 active, 3 gap (< min_gap=5), 5 active -> one run
        a = np.array([1] * 5 + [0] * 3 + [1] * 5, dtype=bool)
        runs = activity_runs(a, min_frames=3, min_gap=5)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0], (0, 13))

    def test_long_gap_splits(self):
        a = np.array([1] * 5 + [0] * 8 + [1] * 5, dtype=bool)
        runs = activity_runs(a, min_frames=3, min_gap=5)
        self.assertEqual(len(runs), 2)

    def test_short_run_dropped(self):
        a = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)  # 2 < min_frames=3
        runs = activity_runs(a, min_frames=3, min_gap=3)
        self.assertEqual(len(runs), 0)


if __name__ == "__main__":
    unittest.main()
