"""Tests: donor selection for Track-C caller-swapped stimuli (§46 Ph3).

Mechanics only: call-type filtering, duration/fingerprint acoustic matching,
mismatch-spike detection, stdlib-csv output. Synthetic arrays; no checkpoints.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.ingestion.donor_selection import (
    DonorCandidate,
    detect_mismatch_spikes,
    select_donor_candidates,
)


def _v(seed):
    rng = np.random.RandomState(seed)
    v = rng.randn(44)
    return v / np.linalg.norm(v)


class TestSelectDonorCandidates(unittest.TestCase):
    def _pool(self):
        return [
            DonorCandidate(call_type="phee", duration_ms=500.0, fingerprint=_v(0), donor_id=7),
            DonorCandidate(call_type="trill", duration_ms=480.0, fingerprint=_v(1), donor_id=8),
            DonorCandidate(call_type="phee", duration_ms=900.0, fingerprint=_v(2), donor_id=9),
            DonorCandidate(
                call_type="phee", duration_ms=510.0, fingerprint=_v(0) * 0.99, donor_id=10
            ),
        ]

    def test_coarse_call_type_filtering(self):
        target = DonorCandidate(call_type="phee", duration_ms=500.0, fingerprint=_v(0), donor_id=0)
        out = select_donor_candidates(target, self._pool(), call_type="phee")
        self.assertTrue(all(c.call_type == "phee" for c in out))
        self.assertNotIn(8, [c.donor_id for c in out])

    def test_duration_and_acoustic_matching(self):
        target = DonorCandidate(call_type="phee", duration_ms=500.0, fingerprint=_v(0), donor_id=0)
        out = select_donor_candidates(
            target,
            self._pool(),
            call_type="phee",
            duration_tolerance_ms=50.0,
            min_cosine=0.9,
        )
        ids = [c.donor_id for c in out]
        self.assertIn(7, ids)  # exact match
        self.assertIn(10, ids)  # near-duplicate fingerprint, within duration
        self.assertNotIn(9, ids)  # duration 900ms >> tolerance
        self.assertNotIn(8, ids)  # wrong call type


class TestDetectMismatchSpikes(unittest.TestCase):
    def test_top_n_spikes_detected_and_csv_written(self):
        delta = np.zeros(600)  # 60 s at 100 ms windows for the test
        delta[100] = 5.0
        delta[250] = 8.0
        delta[400] = 3.0
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "spikes.csv"
            spikes = detect_mismatch_spikes(
                delta,
                hop_ms=100.0,
                top_n=2,
                donor_id=3,
                recipient_id=1,
                out_path=out,
            )
            self.assertEqual(len(spikes), 2)
            self.assertEqual(spikes[0]["start_ms"], 25000.0)  # largest first
            self.assertEqual(spikes[1]["start_ms"], 10000.0)
            with open(out, newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(
                list(rows[0].keys()),
                ["start_ms", "end_ms", "delta_L", "donor_id", "recipient_id"],
            )
            self.assertEqual(float(rows[0]["delta_L"]), 8.0)
            self.assertEqual(int(rows[0]["donor_id"]), 3)

    def test_no_out_path_returns_list_only(self):
        spikes = detect_mismatch_spikes(np.array([0.0, 1.0, 0.0]), hop_ms=10.0, top_n=1)
        self.assertEqual(len(spikes), 1)
        self.assertEqual(spikes[0]["end_ms"] - spikes[0]["start_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()
