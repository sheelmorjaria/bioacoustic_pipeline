"""Tests: Track A yardstick primitives (diagnostics.probes).

No checkpoints, no network. ``forced_align_words`` (wav2vec2) is exercised only
by the harness smoke run, not here.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.diagnostics.probes import (
    linear_probe,
    load_librispeech_transcript,
    match_span_to_label,
    token_label_consistency,
)


class TestLinearProbe(unittest.TestCase):
    def test_recovers_separable_signal(self):
        from sklearn.datasets import make_blobs

        X, y = make_blobs(
            n_samples=[40] * 5, centers=[(0, 0), (12, 0), (0, 12), (12, 12), (24, 24)],
            n_features=8, random_state=0, cluster_std=0.5,
        )
        r = linear_probe(X, y)
        self.assertIsNone(r.skipped)
        self.assertGreater(r.accuracy, 0.9)
        self.assertGreater(r.lift_over_chance, 3.0)
        self.assertTrue(r.signal_exists)
        self.assertEqual(r.n_classes, 5)

    def test_skips_insufficient_data(self):
        # 2 classes, 3 samples each -> <5 classes and <100 rows
        X = np.random.RandomState(0).randn(6, 4)
        y = np.array([0, 0, 0, 1, 1, 1])
        r = linear_probe(X, y)
        self.assertIsNotNone(r.skipped)


class TestTokenLabelConsistency(unittest.TestCase):
    def test_perfect_mapping(self):
        tokens = [0, 0, 0, 1, 1, 1]
        words = ["A", "A", "A", "B", "B", "B"]
        r = token_label_consistency(tokens, words)
        self.assertIsNone(r.skipped)
        self.assertAlmostEqual(r.consistency, 1.0)
        self.assertEqual(r.n_labeled, 6)
        self.assertEqual(r.n_unique_tokens, 2)
        self.assertAlmostEqual(r.chance_baseline, 0.5)

    def test_ignores_unknown(self):
        # one <unknown> entry must be dropped from the labeled count
        tokens = [0, 0, 0, 1, 1]
        words = ["A", "A", "A", "B", "<unknown>"]
        r = token_label_consistency(tokens, words)
        self.assertEqual(r.n_labeled, 4)  # 3 A's + 1 B
        self.assertAlmostEqual(r.consistency, 1.0)

    def test_mixed_purity(self):
        # token 0 -> [A,A,B] (purity 2/3 >= 0.5 -> counts 2); token 1 -> [B,B] (2)
        tokens = [0, 0, 0, 1, 1]
        words = ["A", "A", "B", "B", "B"]
        r = token_label_consistency(tokens, words)
        self.assertAlmostEqual(r.consistency, 0.8)  # (2+2)/5

    def test_no_labeled_segments(self):
        r = token_label_consistency([0, 1], ["<unknown>", "<unknown>"])
        self.assertIsNotNone(r.skipped)


class TestMatchSpanToLabel(unittest.TestCase):
    spans = [(0.0, 100.0, "HELLO"), (100.0, 200.0, "WORLD")]

    def test_picks_overlapping_word(self):
        self.assertEqual(match_span_to_label(40.0, 60.0, self.spans), "HELLO")

    def test_returns_none_beyond_tolerance(self):
        self.assertIsNone(match_span_to_label(1000.0, 1100.0, self.spans))


class TestLoadLibrispeechTranscript(unittest.TestCase):
    def test_parses_trans_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "188-136483-0000.flac").write_bytes(b"")
            (d / "188-136483.trans.txt").write_text(
                "188-136483-0000 HELLO WORLD\n188-136483-0001 ANOTHER LINE\n"
            )
            self.assertEqual(
                load_librispeech_transcript(d / "188-136483-0000.flac"),
                "HELLO WORLD",
            )

    def test_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_librispeech_transcript(Path(tmp) / "x-1-2.flac"), "")


if __name__ == "__main__":
    unittest.main()
