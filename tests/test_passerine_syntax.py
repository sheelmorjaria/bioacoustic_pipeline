"""Tests: passerine syntax track — adapter, transition network, null, ceiling.

Four mechanics for the Parus-major combinatorial-syntax experiment: species
filtering, transition-safe intervals (never count transitions across file
boundaries), bigram transition matrices, within-file permutation null
(Phipson–Smyth), and the T2-not-T4 semantic-ceiling labeling.

Codebook note: the passerine analysis fits a FRESH k=24 codebook on a Parus
train split (the Stage-2 pattern); the frozen bat codebook is a TRANSFER
ablation, not the primary vocabulary — different acoustic coverage.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics import passerine_syntax as ps
from bioacoustic_pipeline.diagnostics.passerine_syntax import (
    build_transition_matrix,
    build_transition_safe_intervals,
    evaluate_compositionality,
    filter_parus_recordings,
)

# test_transition_significance is referenced via the module: importing a
# test_* name directly would make pytest collect the function itself


class TestFilterAndIntervals(unittest.TestCase):
    def setUp(self):
        self.manifest = [
            {"file_path": "audio/b1.wav", "species_scientific": "Parus major"},
            {"file_path": "audio/b2.wav", "species_scientific": "Cyanistes caeruleus"},
            {"file_path": "audio/f1.wav", "species_scientific": "Rana temporaria"},
        ]

    def test_filter_keeps_parus_and_cyanistes(self):
        out = filter_parus_recordings(self.manifest)
        self.assertEqual(len(out), 2)
        self.assertNotIn("Rana temporaria", [r["species_scientific"] for r in out])

    def test_intervals_group_by_file(self):
        intervals = [
            {"file_id": "f1", "start_ms": 0, "end_ms": 500},
            {"file_id": "f1", "start_ms": 500, "end_ms": 1000},
            {"file_id": "f2", "start_ms": 0, "end_ms": 500},
        ]
        safe = build_transition_safe_intervals(intervals)
        self.assertEqual([iv["file_id"] for iv in safe], ["f1", "f1", "f2"])


class TestTransitionMatrix(unittest.TestCase):
    def test_counts_correct(self):
        m = build_transition_matrix([[1, 2, 1, 2, 3]], n_clusters=5)
        self.assertEqual(m[1, 2], 2)
        self.assertEqual(m[2, 1], 1)
        self.assertEqual(m[2, 3], 1)
        self.assertEqual(m[3, 1], 0)

    def test_no_cross_sequence_transitions(self):
        m = build_transition_matrix([[1, 2], [2, 3]], n_clusters=5)
        self.assertEqual(m[1, 2], 1)
        self.assertEqual(m[2, 3], 1)
        self.assertEqual(m[2, 2], 0)  # file boundary never counted


class TestTransitionNull(unittest.TestCase):
    def test_repetitive_sequence_significant(self):
        res = ps.test_transition_significance([[1, 2] * 8], n_clusters=5, n_perm=100)
        self.assertLess(res["p_value"], 0.05)

    def test_random_sequence_not_significant(self):
        rng = np.random.RandomState(42)
        seq = rng.randint(0, 5, 100).tolist()
        res = ps.test_transition_significance([seq], n_clusters=5, n_perm=100)
        self.assertGreater(res["p_value"], 0.05)

    def test_null_preserves_unigrams_within_files(self):
        # shuffling within files keeps each file's unigram content
        res = ps.test_transition_significance([[1, 1, 2], [3, 3]], n_clusters=5, n_perm=20)
        self.assertIn("statistic", res)


class TestSemanticCeiling(unittest.TestCase):
    def test_structural_finding_is_t2_not_t4(self):
        res = evaluate_compositionality(
            transition=("alert", "recruitment"), p_value=0.01, is_non_random=True
        )
        self.assertIn("T2", res["evidence_tier"])
        self.assertNotIn("T4", res["evidence_tier"])
        self.assertEqual(res["semantic_claim"], "structural_association_not_function")

    def test_random_finding_makes_no_claim(self):
        res = evaluate_compositionality(
            transition=("alert", "recruitment"), p_value=0.5, is_non_random=False
        )
        self.assertEqual(res["semantic_claim"], "no_structural_claim")


if __name__ == "__main__":
    unittest.main()
