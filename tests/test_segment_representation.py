"""Segment-representation source + head compatibility tests (offline, torch-free).

Covers: per-species ``resolve_source`` correctness + firewall legality, ``assert_compatible``
match/mismatch/legacy-skip, ``build_backbone`` for an all-species source, the fusion
``NotImplementedError`` stub, head ``input_source`` declarations, and the
``run_pipeline(segment_source=...)`` hook (trace key + mismatch raises).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import unittest

import numpy as np

from bioacoustic_pipeline.embeddings.firewall import ProvenanceViolation
from bioacoustic_pipeline.embeddings.segment_representation import (
    SOURCE_CATEGORIES,
    assert_compatible,
    assert_pairwise_compatible,
    build_backbone,
    resolve_source,
)


class TestResolveSource(unittest.TestCase):
    def test_cpc_resolves_per_species(self):
        human = resolve_source("human", "cpc")
        self.assertEqual(human.provenance, "human_cpc_phonetic")
        self.assertEqual((human.category, human.dim, human.view_class), ("cpc", 256, "learned"))
        bat = resolve_source("bat", "cpc")
        self.assertEqual(bat.provenance, "bat_cpc")

    def test_all_species_sources_any_species(self):
        for sp in ("human", "bat", "marmoset"):
            s = resolve_source(sp, "acoustic_fingerprint")
            self.assertEqual(s.provenance, "acoustic_fingerprint")
            self.assertEqual(s.view_class, "interpretable")

    def test_firewall_rejects_biomae_off_bat(self):
        with self.assertRaises(ProvenanceViolation):
            resolve_source("human", "biomae")  # biomae is bat-only

    def test_unknown_species_cpc_raises(self):
        with self.assertRaises(ValueError):
            resolve_source("dolphin", "cpc")  # no registered species CPC

    def test_biomae_bat_legal(self):
        s = resolve_source("bat", "biomae")
        self.assertEqual((s.provenance, s.dim), ("biomae", 112))


class _MockHead:
    def __init__(self, input_source):
        self.input_source = input_source


class TestAssertCompatible(unittest.TestCase):
    def test_matched_passes(self):
        src = resolve_source("human", "cpc")
        assert_compatible(_MockHead("cpc"), src, head_kind="vq_vae")  # no raise

    def test_mismatched_raises(self):
        src = resolve_source("human", "cpc")
        with self.assertRaises(ValueError):
            assert_compatible(_MockHead("acoustic_fingerprint"), src, head_kind="vq_vae")

    def test_legacy_head_without_input_source_skipped(self):
        src = resolve_source("human", "cpc")
        assert_compatible(object(), src)  # no input_source attr -> silently skipped

    def test_pairwise_compatible(self):
        a = resolve_source("human", "cpc")
        b = resolve_source("bat", "cpc")
        assert_pairwise_compatible(a, b)  # same category
        with self.assertRaises(ValueError):
            assert_pairwise_compatible(a, resolve_source("human", "acoustic_fingerprint"))


class TestBuildBackbone(unittest.TestCase):
    def test_acoustic_fingerprint_all_species_no_torch(self):
        for sp in ("human", "bat"):
            bb = build_backbone(resolve_source(sp, "acoustic_fingerprint"))
            self.assertTrue(bb.available, f"{sp}: {getattr(bb, 'load_error', None)}")
            self.assertEqual(bb.view_class, "interpretable")

    def test_fusion_not_implemented(self):
        # Construct the source directly: resolve_source firewall-rejects the unregistered
        # fusion provenance; build_backbone is the honest stub that raises NotImplementedError.
        from bioacoustic_pipeline.embeddings.segment_representation import SegmentSource

        src = SegmentSource(
            category="embedding_bank_fusion",
            provenance="embedding_bank_fusion",
            species="human",
            dim=None,
            view_class="fused",
        )
        with self.assertRaises(NotImplementedError):
            build_backbone(src)


class TestHeadInputSource(unittest.TestCase):
    def test_cpc_heads_declare_cpc(self):
        # torch-free class-attr check (torch only loads on __init__).
        from bioacoustic_pipeline.tokenization.multiview import AcousticTokenizer

        self.assertEqual(AcousticTokenizer.input_source, "cpc")

    def test_fingerprint_adapter_declares_fingerprint(self):
        from bioacoustic_pipeline.tokenization.fingerprint_tokenizer_adapter import (
            FingerprintTokenizerAdapter,
        )

        self.assertEqual(FingerprintTokenizerAdapter.input_source, "acoustic_fingerprint")

    def test_categories_complete(self):
        self.assertEqual(
            set(SOURCE_CATEGORIES),
            {"cpc", "biomae", "acoustic_fingerprint", "spectrogram", "embedding_bank_fusion"},
        )


class TestRunPipelineHook(unittest.TestCase):
    def test_trace_records_matching_source(self):
        from bioacoustic_pipeline.orchestration import run_pipeline
        from bioacoustic_pipeline.verdict_engine import is_valid

        src = resolve_source("human", "cpc")
        report, trace = run_pipeline([], "human", segment_source=src)
        self.assertTrue(is_valid(report))
        self.assertEqual(trace["segment_representation"]["category"], "cpc")
        self.assertEqual(trace["segment_representation"]["provenance"], "human_cpc_phonetic")

    def test_no_source_keeps_trace_none(self):
        from bioacoustic_pipeline.orchestration import run_pipeline

        _, trace = run_pipeline([], "human")
        self.assertIsNone(trace["segment_representation"])

    def test_mismatched_source_raises(self):
        # A fingerprint tokenizer (input_source='acoustic_fingerprint') vs a cpc segment source.
        from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.tokenizer import (
            FingerprintTokenizer,
        )
        from bioacoustic_pipeline.orchestration import run_pipeline
        from bioacoustic_pipeline.tokenization.fingerprint_tokenizer_adapter import (
            FingerprintTokenizerAdapter,
        )

        fitted = FingerprintTokenizer.fit(np.random.RandomState(0).randn(80, 43), k=8, seed=0)
        tokenizer = FingerprintTokenizerAdapter(fitted, species="human")
        cpc_src = resolve_source("human", "cpc")
        with self.assertRaises(ValueError):
            run_pipeline([], "human", tokenizer=tokenizer, segment_source=cpc_src)


if __name__ == "__main__":
    unittest.main()
