"""Domain-gate + ESC-50 negative-control tests (offline, torch-free).

Covers: clip-source classification, the gate decision, the additive `not_applicable`
schema integration (5 ethological axes refused, 3 acoustic axes still computed), the
dataset summary, the negative-control report builder, and a regression guard that the
no-gate path is byte-unchanged.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics.domain_gate import (
    BIOACOUSTIC_COMMUNICATION_CANDIDATE,
    BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK,
    BIOLOGICAL_SOURCE,
    CANDIDATE,
    DISABLED_AXES,
    NON_BIOACOUSTIC_COMMUNICATION,
    NON_BIOLOGICAL_SOUND,
    NON_BIOLOGICAL_SOURCE,
    REJECTED,
    classify_source,
    gate_clip,
    gate_dataset,
    summarize_dataset,
)
from bioacoustic_pipeline.diagnostics.negative_control_report import (
    build_negative_control_report,
)
from bioacoustic_pipeline.verdict_engine import build_report, is_valid
from bioacoustic_pipeline.verdict_engine.schema import AXIS_LEVELS


class TestClassifySource(unittest.TestCase):
    def test_nonbiological_labels(self):
        for label in ("rain", "sea_waves", "engine", "siren", "wind", "train"):
            self.assertEqual(classify_source(label), NON_BIOLOGICAL_SOURCE, label)

    def test_biological_labels(self):
        for label in ("dog", "rooster", "crickets", "chirping_birds", "frog", "cat"):
            self.assertEqual(classify_source(label), BIOLOGICAL_SOURCE, label)

    def test_space_normalization(self):
        # "chirping birds" (with a space) must normalize to the biological set
        self.assertEqual(classify_source("chirping birds"), BIOLOGICAL_SOURCE)
        self.assertEqual(classify_source("Chirping Birds"), BIOLOGICAL_SOURCE)

    def test_missing_label_ambiguous(self):
        self.assertEqual(classify_source(None), "AMBIGUOUS_SOURCE")


class TestGateDecision(unittest.TestCase):
    def test_esc50_nonbiological_clip(self):
        g = gate_clip("esc-50", "rain")
        self.assertEqual(g.communication_status, REJECTED)
        self.assertEqual(g.verdict, NON_BIOLOGICAL_SOUND)
        self.assertEqual(g.dataset_verdict, NON_BIOACOUSTIC_COMMUNICATION)
        self.assertEqual(set(g.disabled_axes), set(DISABLED_AXES))

    def test_esc50_biological_source_clip_is_not_forced_nonbiological(self):
        # dog is biological source -> BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK, NOT NON_BIOLOGICAL_SOUND
        g = gate_clip("esc-50", "dog")
        self.assertEqual(g.source_type, BIOLOGICAL_SOURCE)
        self.assertEqual(g.verdict, BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK)
        self.assertEqual(g.communication_status, REJECTED)

    def test_dataset_level_rejection(self):
        g = gate_dataset("esc-50")
        self.assertEqual(g.dataset_verdict, NON_BIOACOUSTIC_COMMUNICATION)
        self.assertTrue(g.rejected)

    def test_unknown_dataset_is_candidate(self):
        # a real (annotated) communication corpus is not rejected; no axes disabled
        g = gate_clip("librispeech", label=None)
        self.assertEqual(g.communication_status, CANDIDATE)
        self.assertEqual(g.verdict, BIOACOUSTIC_COMMUNICATION_CANDIDATE)
        self.assertEqual(g.disabled_axes, set())

    def test_summarize_dataset(self):
        results = [gate_clip("esc-50", lab) for lab in ("rain", "engine", "dog", "crickets")]
        s = summarize_dataset(results)
        self.assertEqual(s["n_files"], 4)
        self.assertEqual(s["n_nonbiological_source"], 2)
        self.assertEqual(s["n_biological_source_noncommunication"], 2)
        self.assertEqual(s["communication_candidate_files"], 0)
        self.assertEqual(s["dataset_verdict"], NON_BIOACOUSTIC_COMMUNICATION)


class TestSchemaIntegration(unittest.TestCase):
    def test_not_applicable_is_a_valid_level_for_ethological_axes(self):
        for ax in DISABLED_AXES:
            self.assertIn("not_applicable", AXIS_LEVELS[ax], ax)
        # acoustic axes must NOT carry not_applicable (they are always computed)
        for ax in ("sequence_recombination", "call_unitization", "acoustic_coherence"):
            self.assertNotIn("not_applicable", AXIS_LEVELS[ax], ax)

    def test_build_report_with_gate_is_schema_valid_and_abstains_bio_axes(self):
        g = gate_clip("esc-50", "dog")
        r = build_report("none", lz_normalized=0.5, n_hard=10, n_soft=2, domain_gate=g)
        self.assertTrue(is_valid(r))
        for ax in DISABLED_AXES:
            self.assertEqual(r["axes"][ax]["level"], "not_applicable", ax)
        for ax in ("sequence_recombination", "call_unitization", "acoustic_coherence"):
            self.assertNotEqual(r["axes"][ax]["level"], "not_applicable", ax)
        self.assertEqual(r["domain"]["verdict"], NON_BIOACOUSTIC_COMMUNICATION)
        # overall_confidence excludes not_applicable axes (it's > 0, not dragged to ~0)
        self.assertGreater(r["overall_confidence"], 0.0)

    def test_no_gate_path_is_unchanged(self):
        r = build_report("human", lz_normalized=0.5, n_hard=10, n_soft=2)
        self.assertTrue(is_valid(r))
        self.assertNotIn("domain", r)  # no domain block on an un-gated run
        self.assertEqual(r["axes"]["semantic_validation"]["level"], "none")
        # no axis is refused on the un-gated path
        for ax in DISABLED_AXES:
            self.assertNotEqual(r["axes"][ax]["level"], "not_applicable", ax)


class TestNegativeControlReport(unittest.TestCase):
    def _clips(self, n=8, sr=16000, seed=0):
        rng = np.random.default_rng(seed)
        labels = ["rain", "engine", "dog", "crickets", "wind", "rooster", "siren", "cat"]
        out = []
        for i in range(n):
            label = labels[i % len(labels)]
            t = np.arange(sr * 2, dtype=np.float64) / sr
            if label in {"dog", "crickets", "rooster", "cat"}:
                x = np.sin(2 * np.pi * 600 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 8 * t))
            else:
                x = rng.standard_normal(sr * 2)
            x[::1000] *= 3.0  # a few onsets
            out.append((x.astype(np.float32), sr, label))
        return out

    def test_report_refuses_communication(self):
        report = build_negative_control_report("esc-50", self._clips(), k=8, seed=0)
        self.assertEqual(report["final_verdict"]["class"], NON_BIOACOUSTIC_COMMUNICATION)
        self.assertEqual(report["domain_gate"]["verdict"], NON_BIOACOUSTIC_COMMUNICATION)
        # biological axes refused
        for ax in DISABLED_AXES:
            self.assertEqual(report["disabled_biological_diagnostics"][ax], "NOT_APPLICABLE")
        # acoustic structure still computed
        self.assertTrue(report["pipeline_outputs"]["nbd_segments"].startswith("computed"))
        self.assertTrue(report["pipeline_outputs"]["vq_tokens"].startswith("computed"))
        # human-only affect deliberately not computed
        self.assertEqual(report["pipeline_outputs"]["beta_vae_affect"], "not_computed_human_only")
        # both source classes present in the synthetic mix
        s = report["summary"]
        self.assertGreater(s["n_nonbiological_source"], 0)
        self.assertGreater(s["n_biological_source_noncommunication"], 0)
        self.assertEqual(s["communication_candidate_files"], 0)


if __name__ == "__main__":
    unittest.main()
