"""Integration: the full chain emits a schema-valid verdict report.

Also smoke-tests the migrated segmentation (proposal generator) on synthetic
audio so the cherry-picked math is wired correctly in the new package.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.null_models import (
    within_file_shuffle,
)
from bioacoustic_pipeline.orchestration import run_report, synthesize_signals
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals
from bioacoustic_pipeline.verdict_engine import validate_report

SR = 16000


def _tone(freq, dur):
    t = np.arange(int(SR * dur)) / SR
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestSegmentationMigrated(unittest.TestCase):
    def test_proposal_generator_runs_in_new_package(self):
        audio = np.concatenate([np.zeros(SR // 2, dtype=np.float32), _tone(1000.0, 0.5)])
        sig = compute_signals(audio, SR)
        props = generate_proposals(sig)
        self.assertTrue(any(abs(p.time_ms - 500.0) < 60 for p in props))


class TestFullChainReport(unittest.TestCase):
    def _fixtures(self):
        rng = np.random.RandomState(0)
        tokens = rng.randint(0, 8, 400).tolist()
        boundary_types = [["hard", "soft", "soft", "transitional"][i % 4] for i in range(400)]
        # per-file sequences for the alias path
        sequences = [rng.randint(0, 8, 10).tolist() for _ in range(40)]
        return tokens, boundary_types, sequences

    def test_report_is_schema_valid(self):
        tokens, bt, seqs = self._fixtures()
        report = run_report("bat", tokens, bt, sequences=seqs)
        self.assertEqual(validate_report(report), [], "full-chain report must satisfy schema")

    def test_report_carries_alias_signal(self):
        tokens, bt, seqs = self._fixtures()
        signals = synthesize_signals(tokens, bt, sequences=seqs)
        self.assertIn("alias_density", signals)
        self.assertGreaterEqual(signals["lz_normalized"], 0.0)

    def test_within_file_shuffle_preserves_unigrams(self):
        from collections import Counter

        seq = [1, 2, 3, 4, 1, 2, 3, 4]
        rng = np.random.RandomState(1)
        shuf = within_file_shuffle([seq], rng)[0]
        self.assertEqual(Counter(seq), Counter(shuf))


if __name__ == "__main__":
    unittest.main()
