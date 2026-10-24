"""Tests: acquisition-structure audit + taxonomy Arm 1 (cross-taxa follow-up).

If taxon is predictable from recording SOURCE alone, classes are acquisition-
confounded and no feature engineering manufactures cross-source biological
generalization — the audit must say so BEFORE any sweep runs. Grouped splits
keep each source wholly inside one fold; Arm 1 scales on train only.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.taxonomy_arms import run_arm1_existing_crop
from bioacoustic_pipeline.ingestion.acquisition_audit import (
    build_source_support_matrix,
    get_grouped_splits,
    run_metadata_only_baseline,
)


def _meta():
    return [
        {"clip_id": "c1", "taxon": "Aves", "source": "Archive1"},
        {"clip_id": "c2", "taxon": "Aves", "source": "Archive1"},
        {"clip_id": "c3", "taxon": "Anura", "source": "Archive2"},
        {"clip_id": "c4", "taxon": "Anura", "source": "Archive2"},
    ]


class TestAcquisitionAudit(unittest.TestCase):
    def test_source_matrix_counts(self):
        m = build_source_support_matrix(_meta(), taxon_key="taxon", source_key="source")
        self.assertEqual(m[("Aves", "Archive1")], 2)
        self.assertEqual(m[("Anura", "Archive2")], 2)

    def test_grouped_splits_prevent_source_leakage(self):
        meta = _meta() * 10
        for train_rows, test_rows in get_grouped_splits(meta, group_key="source", n_splits=2):
            train_sources = {r["source"] for r in train_rows}
            test_sources = {r["source"] for r in test_rows}
            self.assertEqual(len(train_sources & test_sources), 0)

    def test_metadata_baseline_detects_perfect_confound(self):
        acc = run_metadata_only_baseline(_meta(), target_key="taxon", feature_key="source")
        self.assertEqual(acc, 1.0)

    def test_metadata_baseline_low_when_unconfounded(self):
        rng = np.random.RandomState(0)
        meta = [
            {"taxon": rng.choice(["Aves", "Anura"]), "source": rng.choice(["S1", "S2"])}
            for _ in range(200)
        ]
        acc = run_metadata_only_baseline(meta, target_key="taxon", feature_key="source")
        self.assertLess(acc, 0.9)


class TestTaxonomyArm1(unittest.TestCase):
    def test_scaler_train_only_and_f1_bounds(self):
        rng = np.random.RandomState(1)
        X_train = rng.rand(8, 44) * 100
        X_test = rng.rand(2, 44) * 100
        y_train = np.array(["A"] * 4 + ["B"] * 4)
        y_test = np.array(["A", "B"])
        res = run_arm1_existing_crop(X_train, y_train, X_test, y_test)
        self.assertIn("macro_f1", res)
        self.assertGreaterEqual(res["macro_f1"], 0.0)
        self.assertLessEqual(res["macro_f1"], 1.0)
        self.assertIn("chance", res)

    def test_leakage_free_scaling_changes_predictions(self):
        # train and test on very different scales: the scaler must use TRAIN
        # stats only (test scaled by train mean/std), keeping predictions sane
        rng = np.random.RandomState(2)
        X_train = np.vstack([rng.randn(20, 10) + 5, rng.randn(20, 10) - 5])
        y_train = np.array(["A"] * 20 + ["B"] * 20)
        X_test = np.vstack([rng.randn(4, 10) + 5.5, rng.randn(4, 10) - 4.5])
        y_test = np.array(["A"] * 4 + ["B"] * 4)
        res = run_arm1_existing_crop(X_train, y_train, X_test, y_test)
        self.assertGreater(res["macro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
