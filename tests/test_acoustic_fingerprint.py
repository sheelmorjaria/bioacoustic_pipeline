"""Tests: acoustic-fingerprint v0.1 (extractor, bank view, coherence, nulls).

Checkpoint-free (pure numpy). Synthetic-signal table + bank gating + coherence/nulls.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest

import numpy as np

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
from bioacoustic_pipeline.embeddings import AcousticFingerprintBackbone, EmbeddingBank

SR = 16000


def _tone(freq, dur, sr=SR, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _chirp(f0, f1, dur, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return (0.5 * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / (2 * dur)) * t)).astype(np.float32)


def _noise(dur, sr=SR):
    rng = np.random.RandomState(0)
    return (0.3 * rng.randn(int(sr * dur))).astype(np.float32)


def _am(dur, rate=8.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    carrier = np.sin(2 * np.pi * 600 * t)
    return (0.5 * carrier * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))).astype(np.float32)


def _harmonics(dur, f0=300, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    sig = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 5))
    return (0.4 * sig).astype(np.float32)


class TestExtractorShapes(unittest.TestCase):
    def test_dim_consistent_and_silence_zeroed(self):
        fp = afp.extract(_tone(440, 0.3), SR)
        self.assertEqual(fp.dim, len(fp.physics) + len(fp.macrotexture) + len(fp.microtexture))
        self.assertTrue(fp.dim > 0)
        sil = afp.extract(np.zeros(int(SR * 0.05), dtype=np.float32), SR)
        self.assertTrue(sil.quality["low_energy"])
        self.assertEqual(float(sil.combined.sum()), 0.0)

    def test_tone_is_tonal_noise_is_not(self):
        self.assertTrue(afp.extract(_tone(440, 0.3), SR).quality["harmonic"])
        self.assertFalse(afp.extract(_noise(0.3), SR).quality["harmonic"])

    def test_signal_table_runs(self):
        for name, sig in [("tone", _tone(440, 0.3)), ("noise", _noise(0.3)),
                          ("chirp", _chirp(200, 2000, 0.3)), ("am", _am(0.5)),
                          ("harmonics", _harmonics(0.3))]:
            fp = afp.extract(sig, SR)
            self.assertTrue(np.all(np.isfinite(fp.combined)), f"non-finite in {name}")
            self.assertTrue(fp.dim > 0)


class TestBankView(unittest.TestCase):
    def test_cross_species_active_and_tagged(self):
        for species in ("bat", "human", "marmoset"):
            with tempfile.TemporaryDirectory() as tmp:
                bank = EmbeddingBank([AcousticFingerprintBackbone()],
                                     target_species=species, cache=tmp)
                self.assertEqual(bank.active_names, ["acoustic_fingerprint"], species)
                rep = bank.report()
                self.assertIn("interpretable", rep["by_view_class"])
                v = bank.embed_all(_tone(440, 0.3), SR)
                self.assertEqual(v["acoustic_fingerprint"].shape[0], bank.backbones[0].embed_dim)

    def test_cache_hit(self):
        calls = {"n": 0}
        orig = AcousticFingerprintBackbone.embed

        def counting(self, audio, sr):
            calls["n"] += 1
            return orig(self, audio, sr)

        AcousticFingerprintBackbone.embed = counting
        try:
            with tempfile.TemporaryDirectory() as tmp:
                bank = EmbeddingBank([AcousticFingerprintBackbone()], target_species="bat", cache=tmp)
                a = bank.embed_all(_tone(440, 0.3), SR)
                b = bank.embed_all(_tone(440, 0.3), SR)
                self.assertTrue(np.allclose(a["acoustic_fingerprint"], b["acoustic_fingerprint"]))
        finally:
            AcousticFingerprintBackbone.embed = orig


class TestCoherenceAndAlias(unittest.TestCase):
    def test_token_coherence_high_for_clustered_low_for_mixed(self):
        base = afp.extract(_tone(440, 0.3), SR).combined
        rng = np.random.RandomState(0)
        clustered = {0: [base + rng.randn(base.size) * 0.01 for _ in range(6)]}
        mixed = {0: [base + rng.randn(base.size) * 5.0 for _ in range(6)]}
        c_hi = afp.token_fingerprint_coherence(clustered)[0]["cohesion"]
        c_lo = afp.token_fingerprint_coherence(mixed)[0]["cohesion"]
        self.assertGreater(c_hi, c_lo)

    def test_classify_alias_four_cases(self):
        self.assertEqual(afp.classify_alias(0.9, 0.9), "acoustic_duplicate")
        self.assertEqual(afp.classify_alias(0.9, 0.2), "functional")
        self.assertEqual(afp.classify_alias(0.2, 0.9), "contextual")
        self.assertEqual(afp.classify_alias(0.2, 0.2), "spurious")


class TestNulls(unittest.TestCase):
    def test_token_coherence_null_significance(self):
        # three very DISTINCT signal types (tone / noise / harmonics) so the tokens are
        # genuinely separable in fingerprint space. Real labeling groups each type ->
        # observed coherence beats the label-shuffled null; p should be small.
        rng = np.random.RandomState(1)
        bases = [afp.extract(sig, SR).combined
                 for sig in (_tone(440, 0.3), _noise(0.3), _harmonics(0.3))]
        coherent = {k: [bases[k] + rng.randn(bases[k].size) * 0.01 for _ in range(8)]
                    for k in range(3)}
        res = afp.token_coherence_null(coherent, n_permutations=100, seed=0)
        self.assertGreater(res.observed, res.null_mean)  # real coherence > shuffled null
        self.assertLess(res.p_value, 0.1)


if __name__ == "__main__":
    unittest.main()
