"""Tests: caller-supervision provenance flag + enforcement (firewall §45).

The caller-conditioned CPC is supervised metric learning, not self-supervised
representation: it can never be *evidence* for caller-structure claims
(circularity), so the registry carries an explicit supervision flag and the
firewall forbids verdict_weight/tokenize for such models REGARDLESS of a
(registry-edit) allowed_use — belt and braces, a bad config edit cannot
re-authorize what supervision semantics forbid.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.embeddings.firewall import Firewall, ProvenanceViolation, Use


class TestCallerSupervisionField(unittest.TestCase):
    def test_registry_entries_exist_with_correct_flags(self):
        fw = Firewall()
        v2 = fw.get("bat_cpc_v2")
        self.assertFalse(v2.contains_caller_supervision)
        self.assertIn("bat", v2.applies_to_species)
        cc = fw.get("bat_cpc_caller")
        self.assertTrue(cc.contains_caller_supervision)
        self.assertIn("bat", cc.applies_to_species)

    def test_legacy_json_without_field_defaults_false(self):
        entry = {
            "pretraining_domain": "egyptian_fruit_bat",
            "contains_human_speech": False,
            "contains_text_supervision": False,
            "allowed_use": ["embedding"],
            "forbidden_use": [],
            "applies_to_species": ["bat"],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"models": {"old_model": entry}}, fh)
            path = fh.name
        fw = Firewall(path)
        self.assertFalse(fw.get("old_model").contains_caller_supervision)
        fw.assert_allowed("old_model", Use.EMBEDDING)  # no raise

    def test_audit_exposes_flag(self):
        fw = Firewall()
        audited = {m["model_name"]: m for m in fw.audit()}
        self.assertIn("contains_caller_supervision", audited["bat_cpc_caller"])
        self.assertTrue(audited["bat_cpc_caller"]["contains_caller_supervision"])


class TestCallerSupervisionEnforcement(unittest.TestCase):
    def setUp(self):
        self.fw = Firewall()

    def test_caller_conditioned_allows_probe_and_retrieval(self):
        self.fw.assert_allowed("bat_cpc_caller", Use.PROBE)
        self.fw.assert_allowed("bat_cpc_caller", Use.RETRIEVAL)

    def test_caller_conditioned_blocks_embedding_and_tokenize(self):
        for use in (Use.EMBEDDING, Use.TOKENIZE, Use.VERDICT_WEIGHT):
            with self.assertRaises(ProvenanceViolation):
                self.fw.assert_allowed("bat_cpc_caller", use)

    def test_enforcement_independent_of_allowed_use(self):
        # even a hand-edited registry that adds tokenize to allowed_use must
        # still hard-fail for a caller-supervised model
        raw = json.loads(Path(self.fw.metadata_path).read_text())
        raw["models"]["bat_cpc_caller"]["allowed_use"] = [
            "embedding",
            "tokenize",
            "verdict_weight",
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(raw, fh)
            tampered = fh.name
        fw = Firewall(tampered)
        for use in (Use.TOKENIZE, Use.VERDICT_WEIGHT):
            with self.assertRaises(ProvenanceViolation):
                fw.assert_allowed("bat_cpc_caller", use)

    def test_self_supervised_v2_keeps_embedding_and_tokenize(self):
        self.fw.assert_allowed("bat_cpc_v2", Use.EMBEDDING)
        self.fw.assert_allowed("bat_cpc_v2", Use.TOKENIZE)
        self.fw.assert_allowed("bat_cpc_v2", Use.PROBE)

    def test_cross_species_still_blocks(self):
        with self.assertRaises(ProvenanceViolation):
            self.fw.assert_cross_species("bat_cpc_v2", "marmoset")


class _CallerSupervisedStub:
    name = "bat_cpc_caller"
    provenance_name = "bat_cpc_caller"
    embed_dim = 128

    @property
    def available(self):
        return True

    @property
    def load_error(self):
        return None

    def embed(self, audio, sr):
        return np.ones(self.embed_dim, dtype=np.float32)


class TestBankGatingCallerSupervised(unittest.TestCase):
    def test_caller_supervised_backbone_excluded_from_bank(self):
        from bioacoustic_pipeline.embeddings.bank import EmbeddingBank

        bank = EmbeddingBank([_CallerSupervisedStub()], target_species="bat")
        self.assertEqual(bank.active_names, [])
        self.assertTrue(any(n == "bat_cpc_caller" for n, _ in bank.skipped))


if __name__ == "__main__":
    unittest.main()
