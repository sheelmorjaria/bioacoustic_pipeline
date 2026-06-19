"""Tests: JSON-driven firewall, embedding cache, bank gating (#2)."""

import unittest

import numpy as np

from bioacoustic_pipeline.embeddings.cache import EmbeddingCache, audio_hash
from bioacoustic_pipeline.embeddings.firewall import Firewall, ProvenanceViolation, Use


class TestJsonFirewall(unittest.TestCase):
    def setUp(self):
        self.fw = Firewall()  # auto-resolves configs/model_metadata.json

    def test_registry_loaded_from_json(self):
        for name in ("human_cpc_phonetic", "aves", "naturelm_audio"):
            self.assertIn(name, [m.model_name for m in [self.fw.get(name)]])

    def test_naturelm_blocked_for_embedding(self):
        with self.assertRaises(ProvenanceViolation):
            self.fw.assert_allowed("naturelm_audio", Use.EMBEDDING)

    def test_aves_allowed_for_embedding(self):
        self.fw.assert_allowed("aves", Use.EMBEDDING)  # no raise

    def test_no_model_may_semantic_verdict(self):
        for m in ("aves", "perch_2", "human_cpc_phonetic"):
            with self.assertRaises(ProvenanceViolation):
                self.fw.assert_allowed(m, Use.SEMANTIC_VERDICT)

    def test_cross_species_blocks_human_for_bat(self):
        with self.assertRaises(ProvenanceViolation):
            self.fw.assert_cross_species("human_cpc_phonetic", "bat")
        self.fw.assert_cross_species("aves", "bat")  # multi-taxa ok

    def test_unregistered_model_blocked(self):
        with self.assertRaises(ProvenanceViolation):
            self.fw.assert_allowed("mystery_model", Use.EMBEDDING)


class TestCache(unittest.TestCase):
    def test_hash_is_content_addressed(self):
        a = np.zeros(100, dtype=np.float32)
        b = np.zeros(100, dtype=np.float32)
        b[50] = 1.0
        self.assertEqual(audio_hash(a, 16000), audio_hash(np.zeros(100, dtype=np.float32), 16000))
        self.assertNotEqual(audio_hash(a, 16000), audio_hash(b, 16000))

    def test_embed_caches_and_does_not_recompute(self):
        import tempfile

        calls = {"n": 0}

        def compute(audio, sr):
            calls["n"] += 1
            return np.ones(8)

        with tempfile.TemporaryDirectory() as tmp:
            cache = EmbeddingCache(root=tmp)
            audio = np.arange(100, dtype=np.float32)
            v1 = cache.embed("fake_backbone", audio, 16000, compute)
            v2 = cache.embed("fake_backbone", audio, 16000, compute)
            np.testing.assert_array_equal(v1, v2)
            self.assertEqual(calls["n"], 1, "second embed must be a cache hit")


class _FakeBackbone:
    def __init__(self, name, provenance_name, dim):
        self.name = name
        self.provenance_name = provenance_name
        self.embed_dim = dim

    @property
    def available(self):
        return True

    @property
    def load_error(self):
        return None

    def embed(self, audio, sr):
        return np.ones(self.embed_dim, dtype=np.float32)


class TestBankGating(unittest.TestCase):
    def test_text_supervised_backbone_excluded(self):
        from bioacoustic_pipeline.embeddings.bank import EmbeddingBank

        bank = EmbeddingBank([_FakeBackbone("naturelm", "naturelm_audio", 1024)])
        self.assertEqual(bank.active_names, [])
        self.assertTrue(any(n == "naturelm" for n, _ in bank.skipped))

    def test_cross_species_backbone_excluded(self):
        from bioacoustic_pipeline.embeddings.bank import EmbeddingBank

        bank = EmbeddingBank(
            [_FakeBackbone("cpc", "human_cpc_phonetic", 256)], target_species="bat"
        )
        self.assertEqual(bank.active_names, [])

    def test_active_backbone_embeds_with_cache(self):
        import tempfile

        from bioacoustic_pipeline.embeddings.bank import EmbeddingBank

        with tempfile.TemporaryDirectory() as tmp:
            bank = EmbeddingBank([_FakeBackbone("aves", "aves", 768)], cache=tmp)
            views = bank.embed_all(np.arange(100, dtype=np.float32), 16000)
            self.assertIn("aves", views)
            self.assertEqual(views["aves"].shape, (768,))


if __name__ == "__main__":
    unittest.main()
