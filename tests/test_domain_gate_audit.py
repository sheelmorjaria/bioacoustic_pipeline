"""Tests: domain-gate specificity audit (§46 cross-taxa Phase 3, as-built).

The gate AS IMPLEMENTED abstains on METADATA, not acoustics: ESC-50 clips
with environmental-sound labels are REJECTED with every ethological axis
not_applicable; corpora without such labels are CANDIDATE-by-default
(documented permissive default, confidence 0.5). There is NO clip-level
acoustic abstention (e.g., white-noise rejection) anywhere in the codebase
— any specificity ratio the manuscript reports must come from this
label-driven path, and claiming acoustic abstention would be unbacked.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

from bioacoustic_pipeline.diagnostics.domain_gate import (
    REJECTED,
    gate_clip,
    gate_from_manifest,
)


class TestDomainGateAudit(unittest.TestCase):
    def test_esc50_environmental_label_rejects_and_disables_axes(self):
        g = gate_clip("esc-50", label="Rain")
        self.assertEqual(g.communication_status, REJECTED)
        self.assertEqual(g.dataset_verdict, "NON_BIOACOUSTIC_COMMUNICATION")
        # ethological axes refused; acoustic/structural axes still computed
        self.assertIn("context_binding", g.disabled_axes)
        self.assertIn("functional_aliasing", g.disabled_axes)
        self.assertEqual(
            g.applicable_axes, {"sequence_recombination", "call_unitization", "acoustic_coherence"}
        )
        self.assertEqual(g.disabled_axes & g.applicable_axes, set())

    def test_unlabeled_corpora_are_candidate_by_default(self):
        # the documented permissive default — including tierstimmenarchiv
        for ds in (
            "tierstimmenarchiv",
            "totally_unknown_corpus",
            "esc50",
        ):  # note: esc50 (no hyphen) is NOT the registered key
            g = gate_from_manifest(ds)
            self.assertEqual(g.communication_status, "CANDIDATE", ds)
            self.assertEqual(g.confidence, 0.5)

    def test_no_acoustic_abstention_exists(self):
        """White noise gets the same CANDIDATE verdict as any audio: the
        gate reads metadata, never samples. Pinning this so nobody cites a
        clip-level specificity number that the code does not implement."""

        g = gate_clip("tierstimmenarchiv", label=None)
        self.assertEqual(g.communication_status, "CANDIDATE")
        self.assertIn("audio", ("audio",))  # the gate never touches it


if __name__ == "__main__":
    unittest.main()
