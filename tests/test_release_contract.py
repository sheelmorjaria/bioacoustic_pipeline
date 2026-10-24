"""Fixture tests: the six real failure modes, gated end to end.

Each fixture reproduces a failure actually found in a 2026-10 run and
asserts the release contract refuses to proceed. Source validity, detected
activity, feature support and task eligibility stay separate concepts.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.acoustic_derivatives import FEATURE_SCHEMA_VERSION
from bioacoustic_pipeline.pipeline.release_contract import (
    MASK_KINDS,
    TASK_REPRESENTATION_POLICY,
    MaskProvenance,
    ReleaseContractError,
    ScientificRunContract,
    assert_derivative_support,
    assert_feature_schema,
    assert_mask_disagreement_reported,
    assert_no_duplicate_sources,
    assert_no_eval_leakage,
    assert_no_padding_as_valid,
    raise_for_science,
    require_masked_recurrence,
)


def _provenance(kind):
    return MaskProvenance(
        kind=kind,
        produced_by="fixture",
        win_ms=40.0,
        hop_ms=10.0,
        amplitude_stage="native waveform",
    )


class TestSchemaGates(unittest.TestCase):
    def test_current_schema_passes(self):
        assert_feature_schema(FEATURE_SCHEMA_VERSION)  # no raise

    def test_layout_drift_refused(self):
        with self.assertRaises(ReleaseContractError):
            assert_feature_schema(1)  # v1: env_cv@18 read mod_depth


class TestPaddingFixture(unittest.TestCase):
    def test_padding_marked_valid_refused(self):
        # fixture 1: 72.89% all-zero cached frames — a mask counting them as
        # valid would observe padding/silence as acoustic evidence
        frames = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [3.0, 1.0]])
        bad = np.array([True, True, False, True])
        with self.assertRaises(ReleaseContractError):
            assert_no_padding_as_valid(frames, bad)
        good = np.array([True, False, False, True])
        assert_no_padding_as_valid(frames, good)  # no raise


class TestPartialSupportFixture(unittest.TestCase):
    def test_single_valid_frame_refused_as_trajectory(self):
        # fixture 2: the [1 valid + 39 gated] recurrence windows
        valid = np.zeros(40, dtype=bool)
        valid[0] = True
        with self.assertRaises(ReleaseContractError):
            assert_derivative_support(valid, min_adjacent_pairs=1)
        valid[1] = True
        assert_derivative_support(valid, min_adjacent_pairs=1)  # no raise


class TestMaskDisagreementFixture(unittest.TestCase):
    def test_one_mask_standing_in_for_both_refused(self):
        # fixture 3: peak-relative vocal check standing in for the absolute
        # fingerprint gate (rejected recurrence candidates)
        with self.assertRaises(ReleaseContractError):
            assert_mask_disagreement_reported(
                {"detected_activity": _provenance("detected_activity")}
            )
        summary = assert_mask_disagreement_reported(
            {
                "detected_activity": _provenance("detected_activity"),
                "source_validity": _provenance("source_validity"),
            }
        )
        self.assertIn("detected_activity", summary["declared"])

    def test_four_concepts_distinct(self):
        self.assertEqual(
            sorted(MASK_KINDS),
            [
                "detected_activity",
                "feature_support",
                "source_validity",
                "task_eligibility",
            ],
        )


class TestDuplicateAndLeakageFixtures(unittest.TestCase):
    def test_duplicate_sources_refused(self):
        with self.assertRaises(ReleaseContractError):
            assert_no_duplicate_sources(["a.wav", "b.wav", "a.wav"])

    def test_same_source_both_sides_refused(self):
        with self.assertRaises(ReleaseContractError):
            assert_no_duplicate_sources(["a.wav", "a.wav"], parts=[0, 1])

    def test_eval_leakage_refused(self):
        with self.assertRaises(ReleaseContractError):
            assert_no_eval_leakage(["a.wav", "b.wav"], ["b.wav", "c.wav"])
        assert_no_eval_leakage(["a.wav"], ["c.wav"])  # no raise


class TestRecurrenceMaskingGate(unittest.TestCase):
    def test_unmasked_search_refused(self):
        with self.assertRaises(ReleaseContractError):
            require_masked_recurrence(None, None)


class TestFeatureContractGate(unittest.TestCase):
    """The marmoset replication's 44-D violation must be refused at entry."""

    def test_wide_array_refused(self):
        from bioacoustic_pipeline.pipeline.release_contract import (
            ReleaseContractError,
            assert_feature_contract,
        )

        wide = np.zeros((10, 44))  # 44-D entering a 12-feature study
        with self.assertRaises(ReleaseContractError):
            assert_feature_contract(wide, context="fitting")

    def test_wrong_width_list_refused(self):
        from bioacoustic_pipeline.pipeline.release_contract import (
            ReleaseContractError,
            assert_feature_contract,
        )

        with self.assertRaises(ReleaseContractError):
            assert_feature_contract(["a"] * 13, context="scoring")

    def test_correct_width_passes(self):
        from bioacoustic_pipeline.pipeline.release_contract import (
            assert_feature_contract,
        )

        assert_feature_contract(np.zeros((10, 12)))  # no raise

    def test_stale_version_refused(self):
        from bioacoustic_pipeline.pipeline.release_contract import (
            ReleaseContractError,
            assert_feature_contract,
        )

        with self.assertRaises(ReleaseContractError):
            assert_feature_contract(np.zeros((10, 12)), expected_version=1, context="scoring")


class TestFullContract(unittest.TestCase):
    def _contract(self, **over):
        c = ScientificRunContract(
            run_name="fixture-run",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            masks={
                "detected_activity": _provenance("detected_activity"),
                "source_validity": _provenance("source_validity"),
            },
            source_manifest={
                "a.wav": {"group": "block_0", "part": "fit"},
                "b.wav": {"group": "block_1", "part": "eval"},
            },
            eval_split={"provenance": "fixture split", "unit": "file"},
        )
        for k, v in over.items():
            setattr(c, k, v)
        return c

    def test_valid_contract_passes(self):
        summary = raise_for_science(self._contract())
        self.assertEqual(summary["run"], "fixture-run")
        self.assertIn("caller_id_and_form_retrieval", summary["policy"])

    def test_missing_split_refused(self):
        with self.assertRaises(ReleaseContractError):
            raise_for_science(self._contract(eval_split={}))

    def test_missing_manifest_refused(self):
        with self.assertRaises(ReleaseContractError):
            raise_for_science(self._contract(source_manifest={}))

    def test_leaky_manifest_refused(self):
        from bioacoustic_pipeline.pipeline.release_contract import source_manifest_from

        with self.assertRaises(ReleaseContractError):
            source_manifest_from(
                ["a.wav", "a.wav"], [0, 1], ["block_0", "block_0"]
            )  # same file assembled on both sides

    def test_unmasked_recurrence_mode_refused(self):
        with self.assertRaises(ReleaseContractError):
            raise_for_science(self._contract(recurrence_validity_mode="legacy"))

    def test_policy_covers_all_tasks(self):
        for task, pol in TASK_REPRESENTATION_POLICY.items():
            self.assertIn("default", pol, task)


if __name__ == "__main__":
    unittest.main()
