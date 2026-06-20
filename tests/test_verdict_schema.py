"""Verdict-engine TDD: the assembled report MUST satisfy the canonical schema.

This is the contract every upstream module feeds. Built first per the modular
plan: if the schema is wrong, every module built on it is wrong.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

from bioacoustic_pipeline.verdict_engine import (
    AXIS_NAMES,
    SCHEMA_VERSION,
    build_report,
    format_falsification_report,
    validate_report,
)
from bioacoustic_pipeline.verdict_engine.schema import FALSIFICATION_TRACKS


class TestReportSchema(unittest.TestCase):
    def test_build_report_is_schema_valid(self):
        report = build_report(
            "bat", lz_normalized=0.45, alias_density=0.04, alias_pvalue=0.86,
            n_hard=10, n_soft=1, n_total_boundaries=11,
        )
        errors = validate_report(report)
        self.assertEqual(errors, [], f"schema violations: {errors}")

    def test_report_has_all_six_axes(self):
        report = build_report("human", lz_normalized=0.7, alias_density=0.12, alias_pvalue=0.01)
        self.assertEqual(set(report["axes"]), set(AXIS_NAMES))

    def test_semantics_unvalidated_by_default(self):
        report = build_report("bat", lz_normalized=0.5)
        self.assertFalse(report["validated_semantics"])
        self.assertEqual(report["axes"]["semantic_validation"]["level"], "none")

    def test_falsification_tracks_are_known(self):
        report = build_report("human", lz_normalized=0.7, alias_density=0.12, alias_pvalue=0.01)
        for entry in report["falsification"]:
            for test in entry["falsification_tests"]:
                self.assertIn(test["track"], FALSIFICATION_TRACKS)

    def test_strong_alias_emits_playback_test(self):
        report = build_report("human", lz_normalized=0.7, alias_density=0.20, alias_pvalue=0.01)
        alias_entry = next(e for e in report["falsification"] if e["axis"] == "functional_aliasing")
        labels = " ".join(t["label"] for t in alias_entry["falsification_tests"])
        self.assertIn("playback", labels)
        self.assertTrue(alias_entry["makes_positive_claim"])


class TestEthologicalInvariant(unittest.TestCase):
    """Computation can NEVER assert validated semantics."""

    def test_validated_semantics_without_playback_fails_schema(self):
        report = build_report("bat", lz_normalized=0.5)
        report["validated_semantics"] = True  # forbidden: no Track C evidence
        errors = validate_report(report)
        self.assertTrue(errors, "schema must reject un-validated semantics=True")

    def test_validated_semantics_true_requires_playback_level(self):
        report = build_report("bat", lz_normalized=0.5, semantic_status="playback-supported")
        # engine permits playback-supported -> validated_semantics True; schema accepts.
        self.assertTrue(report["validated_semantics"])
        self.assertEqual(validate_report(report), [])

    def test_behavioral_status_does_not_validate(self):
        # 'behavioral' is partial; only 'playback-supported' validates semantics.
        report = build_report("bat", lz_normalized=0.5, semantic_status="behavioral")
        self.assertFalse(report["validated_semantics"])


class TestFalsificationReportText(unittest.TestCase):
    def test_format_runs_and_has_sections(self):
        from bioacoustic_pipeline.verdict_engine import compute_evidence_profile

        profile = compute_evidence_profile(
            "bat", lz_normalized=0.45, alias_density=0.04, alias_pvalue=0.86,
            n_hard=10, n_soft=1, n_total_boundaries=11,
        )
        text = format_falsification_report(profile)
        self.assertIn("FALSIFICATION REPORT", text)
        self.assertIn("Would be falsified by", text)
        self.assertIn("UNIVERSAL SEMANTICS CAVEAT", text)


class TestMockFixture(unittest.TestCase):
    """The canonical 'final output' mock validates against the schema."""

    def test_mock_final_report_validates(self):
        mock = {
            "schema_version": SCHEMA_VERSION,
            "species": "bat",
            "overall_confidence": 0.6,
            "validated_semantics": False,
            "axes": {
                "sequence_recombination": {"level": "medium", "confidence": 0.6, "evidence": "LZ=0.45"},
                "functional_aliasing": {"level": "absent", "confidence": 0.9, "evidence": "p=0.86"},
                "call_unitization": {"level": "holistic", "confidence": 0.8, "evidence": "10H/1S"},
                "context_binding": {"level": "absent", "confidence": 0.9, "evidence": "0"},
                "turn_structure": {"level": "absent", "confidence": 0.7, "evidence": "0"},
                "acoustic_coherence": {"level": "absent", "confidence": 0.9, "evidence": "no fp"},
                "motif_structure": {"level": "absent", "confidence": 0.9, "evidence": "no motifs"},
                "semantic_validation": {"level": "none", "confidence": 1.0, "evidence": "bio only"},
            },
            "falsification": [
                {"axis": "sequence_recombination", "level": "medium", "confidence": 0.6,
                 "makes_positive_claim": True,
                 "falsification_tests": [{"label": "LZ <= null", "track": "null"}]},
            ],
        }
        self.assertEqual(validate_report(mock), [])


if __name__ == "__main__":
    unittest.main()
