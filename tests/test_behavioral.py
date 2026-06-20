"""Behavioral tests — edge cases, invariants, and integration paths.

Tests WHAT the system does (behavior), not HOW (implementation). Covers:
  - NBD fire-on-drop (spectral_drop signal + floor_drop proposals)
  - Runtime device manager (auto/cpu/cuda + module capability plan)
  - Spectrogram factory (STFT correctness, log-mel, quality, species presets)
  - DualStreamSegment (construction, typed event, grouping)
  - Dual-stream diagnostics (affect variance, token-affect MI, alias-affect, motifs)
  - Null models (permutation test, shuffle strategies preserve marginals)
  - LZ complexity
  - Edge cases (empty, silence, very short, degenerate)

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import unittest

import numpy as np


# --------------------------------------------------------------------------- #
# NBD fire-on-drop
# --------------------------------------------------------------------------- #


class TestNBDFireOnDrop(unittest.TestCase):
    """Spectral drop signal + floor_drop proposal generation."""

    def _make_signals(self, sr=16000, dur_s=0.5):
        from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals

        t = np.arange(int(sr * dur_s)) / sr
        # tone burst: 200ms silence → 200ms tone → 100ms silence (clear rise + drop)
        audio = np.zeros(int(sr * dur_s))
        audio[int(sr * 0.2) : int(sr * 0.4)] = 0.5 * np.sin(2 * np.pi * 440 * t[: int(sr * 0.2)])
        return compute_signals(audio, sr)

    def test_spectral_drop_exists_and_nonneg(self):
        sig = self._make_signals()
        self.assertTrue(hasattr(sig, "spectral_drop"))
        self.assertGreaterEqual(sig.spectral_drop.min(), 0.0)

    def test_spectral_drop_normalised(self):
        sig = self._make_signals()
        self.assertLessEqual(sig.spectral_drop.max(), 1.0 + 1e-9)

    def test_floor_drop_produces_more_proposals(self):
        from bioacoustic_pipeline.segmentation.nbd_proposal import generate_proposals

        sig = self._make_signals()
        without_drop = generate_proposals(sig, floor_energy=0.05, floor_flux=0.05, debounce_ms=30)
        with_drop = generate_proposals(
            sig, floor_energy=0.05, floor_flux=0.05, floor_drop=0.05, debounce_ms=30
        )
        self.assertGreaterEqual(len(with_drop), len(without_drop))

    def test_floor_drop_inf_means_disabled(self):
        from bioacoustic_pipeline.segmentation.nbd_proposal import generate_proposals

        sig = self._make_signals()
        disabled = generate_proposals(sig, floor_drop=float("inf"))
        enabled = generate_proposals(sig, floor_drop=0.01)
        # inf disables => should have fewer-or-equal proposals
        self.assertLessEqual(len(disabled), len(enabled))

    def test_proposal_has_spectral_drop_field(self):
        from bioacoustic_pipeline.segmentation.nbd_proposal import generate_proposals

        sig = self._make_signals()
        props = generate_proposals(sig, floor_energy=0.02, floor_flux=0.02, floor_drop=0.02)
        self.assertGreater(len(props), 0)
        self.assertTrue(hasattr(props[0], "spectral_drop"))
        self.assertGreaterEqual(props[0].spectral_drop, 0.0)


# --------------------------------------------------------------------------- #
# Runtime device manager
# --------------------------------------------------------------------------- #


class TestRuntimeDevice(unittest.TestCase):
    """Device resolution + module capability registry."""

    def test_resolve_cpu(self):
        from bioacoustic_pipeline.runtime import resolve_device

        cfg = resolve_device("cpu")
        self.assertEqual(cfg.device, "cpu")
        self.assertFalse(cfg.is_gpu)
        self.assertFalse(cfg.use_amp)

    def test_resolve_auto_returns_valid_device(self):
        from bioacoustic_pipeline.runtime import resolve_device

        cfg = resolve_device("auto")
        self.assertIn(cfg.device, ("cuda", "mps", "cpu"))
        self.assertIsInstance(cfg.num_workers, int)
        self.assertGreater(cfg.num_workers, 0)

    def test_module_plan_routes_correctly(self):
        from bioacoustic_pipeline.runtime import module_plan

        plan_cpu = module_plan("cpu")
        self.assertEqual(plan_cpu["cpc"], "cpu")
        self.assertEqual(plan_cpu["nbd"], "cpu")
        plan_cuda = module_plan("cuda")
        self.assertEqual(plan_cuda["cpc"], "cuda")
        self.assertEqual(plan_cuda["nbd"], "cpu")
        self.assertEqual(plan_cuda["aves"], "cuda")
        self.assertEqual(plan_cuda["verdict_engine"], "cpu")

    def test_device_config_serializable(self):
        from bioacoustic_pipeline.runtime import resolve_device

        cfg = resolve_device("cpu")
        d = cfg.as_dict()
        self.assertIn("device", d)
        self.assertIn("is_gpu", d)
        self.assertFalse(d["is_gpu"])


# --------------------------------------------------------------------------- #
# Spectrogram factory
# --------------------------------------------------------------------------- #


class TestSpectrogramFactory(unittest.TestCase):
    """STFT, log-mel, quality, species presets."""

    def test_stft_tone_peaks_at_correct_freq(self):
        from bioacoustic_pipeline.representation.spectrograms import compute_stft

        sr = 16000
        t = np.arange(sr // 10) / sr  # 100ms
        x = 0.5 * np.sin(2 * np.pi * 2000 * t)
        mag, freqs, times = compute_stft(x, sr, n_fft=512, hop_length=128)
        peak_freq = freqs[np.argmax(mag.mean(axis=1))]
        self.assertAlmostEqual(peak_freq, 2000, delta=100)

    def test_log_mel_shape(self):
        from bioacoustic_pipeline.representation.spectrograms import compute_log_mel

        sr = 16000
        x = 0.3 * np.random.randn(sr // 2).astype(np.float32)
        mel = compute_log_mel(x, sr, n_mels=64, fmin=0, fmax=8000, n_fft=512, hop_length=128)
        self.assertEqual(mel.shape[0], 64)
        self.assertGreater(mel.shape[1], 0)

    def test_silence_yields_low_energy(self):
        from bioacoustic_pipeline.representation.spectrograms import (
            build_segment_spectrograms,
            get_config,
        )

        cfg = get_config("human")
        sr = cfg.sample_rate
        spec = build_segment_spectrograms(np.zeros(sr // 10), sr, cfg)
        self.assertTrue(spec.quality["low_energy"] or spec.quality["silence"])

    def test_species_pressets(self):
        from bioacoustic_pipeline.representation.spectrograms import SPECIES_PRESETS

        self.assertIn("bat", SPECIES_PRESETS)
        self.assertIn("marmoset", SPECIES_PRESETS)
        self.assertGreater(SPECIES_PRESETS["bat"].fmax, 50000)  # ultrasonic
        self.assertLess(SPECIES_PRESETS["human"].fmax, 10000)

    def test_build_segment_spectrograms_duration(self):
        from bioacoustic_pipeline.representation.spectrograms import (
            build_segment_spectrograms,
            get_config,
        )

        cfg = get_config("marmoset")
        sr = cfg.sample_rate
        x = 0.2 * np.random.randn(sr // 4).astype(np.float32)  # 250ms
        spec = build_segment_spectrograms(x, sr, cfg, species="marmoset", start_ms=0, end_ms=250)
        self.assertAlmostEqual(spec.duration_ms, 250.0)
        self.assertEqual(spec.species, "marmoset")


# --------------------------------------------------------------------------- #
# DualStreamSegment
# --------------------------------------------------------------------------- #


class TestDualStreamSegment(unittest.TestCase):
    """Construction, typed event, grouping."""

    def _fake_segment(self, tok, affect, ctx=None):
        from bioacoustic_pipeline.representation.dual_stream import DualStreamSegment

        return DualStreamSegment(
            segment_id=f"s{tok}",
            start_ms=0,
            end_ms=100,
            vq_token=tok,
            affect_mu=np.array(affect, dtype=np.float32),
            affect_intensity=float(np.linalg.norm(affect)),
            emitter="A" if ctx != "B" else "B",
            context=ctx or 1,
        )

    def test_as_event_has_required_fields(self):
        seg = self._fake_segment(5, [0.1, 0.2])
        ev = seg.as_event()
        for key in ("token", "affect", "affect_norm", "emitter", "context"):
            self.assertIn(key, ev)
        self.assertEqual(ev["token"], 5)

    def test_duration_property(self):
        seg = self._fake_segment(0, [0])
        seg.start_ms, seg.end_ms = 100, 350
        self.assertEqual(seg.duration_ms, 250)

    def test_group_by_emitter(self):
        from bioacoustic_pipeline.representation.dual_stream import group_by_field

        segs = [
            self._fake_segment(1, [0.1], "A"),
            self._fake_segment(2, [0.2], "B"),
            self._fake_segment(3, [0.3], "A"),
        ]
        groups = group_by_field(segs, "emitter")
        self.assertEqual(len(groups["A"]), 2)
        self.assertEqual(len(groups["B"]), 1)


# --------------------------------------------------------------------------- #
# Dual-stream diagnostics
# --------------------------------------------------------------------------- #


class TestDualStreamDiagnostics(unittest.TestCase):
    """Affect variance, MI, alias equivalence, motif signatures."""

    def _make_segs(self, n=50, n_tokens=5, affect_dim=8, seed=0):
        from bioacoustic_pipeline.representation.dual_stream import DualStreamSegment

        rng = np.random.RandomState(seed)
        segs = []
        for i in range(n):
            tok = i % n_tokens
            mu = rng.randn(affect_dim).astype(np.float32) * (0.3 if tok < 2 else 1.0)
            segs.append(
                DualStreamSegment(
                    segment_id=f"s{i}",
                    vq_token=tok,
                    affect_mu=mu,
                    affect_intensity=float(np.linalg.norm(mu)),
                )
            )
        return segs

    def test_affect_within_token_variance_returns_per_token(self):
        from bioacoustic_pipeline.diagnostics.dual_stream_diagnostics import (
            affect_within_token_variance,
        )

        segs = self._make_segs()
        result = affect_within_token_variance(segs)
        self.assertGreater(len(result), 0)
        for tok, stats in result.items():
            self.assertIn("n", stats)
            self.assertIn("affect_var_mean", stats)
            self.assertGreater(stats["n"], 0)

    def test_affect_variance_higher_for_high_var_tokens(self):
        from bioacoustic_pipeline.diagnostics.dual_stream_diagnostics import (
            affect_within_token_variance,
        )

        # tokens 0-1 have low affect variance, 2-4 have high
        segs = self._make_segs(n=100, n_tokens=5)
        result = affect_within_token_variance(segs)
        low_token_var = np.mean([result[t]["affect_var_mean"] for t in (0, 1) if t in result])
        high_token_var = np.mean([result[t]["affect_var_mean"] for t in (2, 3, 4) if t in result])
        self.assertGreater(high_token_var, low_token_var)

    def test_token_affect_mi_returns_float(self):
        from bioacoustic_pipeline.diagnostics.dual_stream_diagnostics import (
            token_affect_mutual_information,
        )

        segs = self._make_segs(n=100, n_tokens=5)
        mi = token_affect_mutual_information(segs)
        self.assertIsInstance(mi, float)
        self.assertGreaterEqual(mi, 0.0)
        self.assertLessEqual(mi, 1.0)

    def test_alias_affect_equivalence_classifies(self):
        from bioacoustic_pipeline.diagnostics.dual_stream_diagnostics import (
            alias_affect_equivalence,
        )

        segs = self._make_segs(n=50, n_tokens=5, affect_dim=8)
        groups = [{0, 1}, {2, 3, 4}]
        results = alias_affect_equivalence(groups, segs)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("affect_equivalence", r)
            self.assertIn("interpretation", r)

    def test_motif_affect_signature(self):
        from bioacoustic_pipeline.diagnostics.dual_stream_diagnostics import motif_affect_signature

        segs = self._make_segs(n=100, n_tokens=5)
        sig = motif_affect_signature([0, 1, 2], segs)
        self.assertEqual(sig["pattern"], [0, 1, 2])
        self.assertEqual(len(sig["mean_trajectory"]), 3)
        self.assertIn("affect_slope", sig)

    def test_empty_input_handled_gracefully(self):
        from bioacoustic_pipeline.diagnostics.dual_stream_diagnostics import (
            affect_within_token_variance,
            token_affect_mutual_information,
        )

        self.assertEqual(affect_within_token_variance([]), {})
        self.assertEqual(token_affect_mutual_information([]), 0.0)


# --------------------------------------------------------------------------- #
# Null models
# --------------------------------------------------------------------------- #


class TestNullModels(unittest.TestCase):
    """Permutation test preserves marginals; shuffle strategies."""

    def test_permutation_test_observed_equals_statistic(self):
        from bioacoustic_pipeline.diagnostics.null_models import (
            permutation_test,
            within_file_shuffle,
        )

        seqs = [[1, 2, 3, 1, 2], [3, 1, 2, 3, 1]]
        stat = lambda s: sum(len(seq) for seq in s)  # noqa: E731
        result = permutation_test(seqs, stat, within_file_shuffle, n_permutations=10)
        self.assertEqual(result.observed, 10.0)  # total tokens
        self.assertGreaterEqual(result.p_value, 0.0)
        # null_type is derived from the shuffle fn name -> "within_file_shuffle"
        self.assertEqual(result.null_type, within_file_shuffle.__name__)

    def test_within_file_shuffle_preserves_per_sequence_lengths(self):
        from bioacoustic_pipeline.diagnostics.null_models import within_file_shuffle

        rng = np.random.RandomState(0)
        seqs = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        shuffled = within_file_shuffle(seqs, rng)
        self.assertEqual([len(s) for s in shuffled], [3, 4, 2])

    def test_cross_file_shuffle_preserves_global_unigram(self):
        from bioacoustic_pipeline.diagnostics.null_models import cross_file_shuffle
        from collections import Counter

        rng = np.random.RandomState(0)
        seqs = [[1, 1, 2], [2, 3, 3]]
        shuffled = cross_file_shuffle(seqs, rng)
        before = sorted(Counter(t for s in seqs for t in s).items())
        after = sorted(Counter(t for s in shuffled for t in s).items())
        self.assertEqual(before, after)


# --------------------------------------------------------------------------- #
# LZ complexity
# --------------------------------------------------------------------------- #


class TestLZComplexity(unittest.TestCase):
    """LZ76 normalised complexity."""

    def test_repetitive_sequence_low_complexity(self):
        from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity

        rep = [1] * 100
        rnd = list(np.random.RandomState(0).randint(0, 10, 100))
        lz_rep = lempel_ziv_complexity(rep).normalized
        lz_rnd = lempel_ziv_complexity(rnd).normalized
        self.assertLess(lz_rep, lz_rnd)

    def test_empty_sequence(self):
        from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity

        result = lempel_ziv_complexity([])
        self.assertGreaterEqual(result.normalized, 0.0)


# --------------------------------------------------------------------------- #
# Edge cases + invariants
# --------------------------------------------------------------------------- #


class TestEdgeCases(unittest.TestCase):
    """Empty input, silence, degenerate data — pipeline must not crash."""

    def test_compute_signals_silence(self):
        from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals

        sig = compute_signals(np.zeros(1600), 16000)
        self.assertGreaterEqual(len(sig.rms), 1)
        self.assertLess(sig.spectral_flux.max(), 1e-6)
        self.assertLess(sig.spectral_drop.max(), 1e-6)

    def test_generate_proposals_empty_audio(self):
        """Silent audio with enough frames to not crash gradient, but zero proposals."""
        from bioacoustic_pipeline.segmentation.nbd_proposal import (
            generate_proposals,
            compute_signals,
        )

        sig = compute_signals(np.zeros(3200), 16000)  # 2+ frames of silence
        props = generate_proposals(sig)
        self.assertEqual(len(props), 0)

    def test_run_pipeline_empty_audio_list(self):
        """run_pipeline must produce a schema-valid report even with zero files."""
        from bioacoustic_pipeline.orchestration import run_pipeline
        from bioacoustic_pipeline.verdict_engine import is_valid

        report, trace = run_pipeline([], "human")
        self.assertTrue(is_valid(report))

    def test_firewall_rejects_cross_species(self):
        from bioacoustic_pipeline.embeddings.firewall import Firewall

        fw = Firewall()
        with self.assertRaises(Exception):
            fw.assert_cross_species("human_vqvae_large", "bat")

    def test_firewall_rejects_unregistered_model(self):
        from bioacoustic_pipeline.embeddings.firewall import Firewall, Use

        fw = Firewall()
        with self.assertRaises(Exception):
            fw.assert_allowed("nonexistent_model", Use.EMBEDDING)

    def test_schema_rejects_not_evaluable_on_non_gated_axis(self):
        """not_evaluable is only valid on sequence-dependent axes."""
        from bioacoustic_pipeline.verdict_engine.schema import AXIS_LEVELS

        # call_unitization should NOT allow not_evaluable
        self.assertNotIn("not_evaluable", AXIS_LEVELS["call_unitization"])

    def test_fingerprint_tokenizer_silent_segment(self):
        """Spectrogram tokenizer returns -1 on all-zero fingerprint."""
        from bioacoustic_pipeline.diagnostics.acoustic_fingerprint.tokenizer import (
            FingerprintTokenizer,
        )

        rng = np.random.RandomState(0)
        tok = FingerprintTokenizer.fit(rng.randn(50, 44), k=4)
        self.assertEqual(tok.tokenize(np.zeros(44)), -1)


if __name__ == "__main__":
    unittest.main()
