"""Tests: raw-audio end-to-end path (orchestration.segment_audio / run_pipeline).

Core tests stay checkpoint-free (a fake tokenizer stands in for the VQ-VAE), so
the suite is green without the ``ml`` extra or any checkpoints. One integration
test exercises the real tokenizer + auto-loaded calibrator and is gated on the
human checkpoints being present.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.orchestration import run_pipeline, segment_audio
from bioacoustic_pipeline.verdict_engine import validate_report

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def _tone(freq, dur, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _bursts(sr=SR, periods=5, period=0.5, freq=440.0):
    """Tone bursts separated by silence — clear energy onsets for the NBD."""
    tone = _tone(freq, period / 2, sr)
    silence = np.zeros(int(sr * period / 2), dtype=np.float32)
    out = np.concatenate([np.concatenate([tone, silence]) for _ in range(periods)])
    return out


def _write_wav(tmp, audio, sr=SR):
    path = str(Path(tmp) / "bursts.wav")
    sf.write(path, audio.astype(np.float32), sr)
    return path


class _FakeTokenizer:
    """Stands in for AcousticTokenizer: emits incrementing token ids."""

    def __init__(self, provenance_name="human_vqvae_large", available=True):
        self.provenance_name = provenance_name
        self._available = available
        self._n = 0

    @property
    def available(self):
        return self._available

    @property
    def load_error(self):
        return None

    def tokenize(self, audio, sr):  # noqa: ARG002
        tok = self._n
        self._n += 1
        return tok


class TestSegmentAudio(unittest.TestCase):
    def test_threshold_fallback_produces_segments(self):
        from bioacoustic_pipeline.ingestion.manifest import load_species_profiles

        profile = load_species_profiles()["human"]
        segs = segment_audio(_bursts(), SR, profile, calibrator=None)
        self.assertGreater(len(segs), 0)
        for seg in segs:
            self.assertIn(seg.boundary_type, {"hard", "soft", "transitional"})
            self.assertLess(seg.start_ms, seg.end_ms)


class TestRunPipelineDegradation(unittest.TestCase):
    def test_schema_valid_with_fake_tokenizer(self):
        audio = _bursts()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(tmp, audio)
            report, trace = run_pipeline(
                [path], "human",
                calibrator=False,  # force fixed-threshold fallback
                tokenizer=_FakeTokenizer(),
            )
        self.assertEqual(validate_report(report), [])
        self.assertTrue(trace["schema_valid"])
        self.assertTrue(trace["tokenizer_available"])
        self.assertGreater(trace["n_tokens"], 0)

    def test_degrades_when_tokenizer_unavailable(self):
        audio = _bursts()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(tmp, audio)
            report, trace = run_pipeline(
                [path], "human",
                calibrator=False,
                tokenizer=_FakeTokenizer(available=False),
            )
        self.assertEqual(validate_report(report), [])
        self.assertFalse(trace["tokenizer_available"])
        self.assertEqual(trace["n_tokens"], 0)
        self.assertIsNone(trace["tokenizer_skipped_reason"])

    def test_firewall_blocks_cross_species_tokenizer(self):
        audio = _bursts()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(tmp, audio)
            report, trace = run_pipeline(
                [path], "bat",
                calibrator=False,
                tokenizer=_FakeTokenizer(provenance_name="human_vqvae_large"),
            )
        # human VQ-VAE may not be applied to bats -> tokenization firewall-skipped
        self.assertEqual(validate_report(report), [])
        self.assertFalse(trace["tokenizer_available"])
        self.assertEqual(trace["n_tokens"], 0)
        self.assertIn("firewall", trace["tokenizer_skipped_reason"])

    def test_unknown_species_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_pipeline([_write_wav(tmp, _bursts())], "unicorn",
                             calibrator=False, tokenizer=_FakeTokenizer())


_HUMAN_CKPT = ROOT / "checkpoints" / "human_vqvae_large" / "best_model.pt"
_HUMAN_CPC = ROOT / "checkpoints" / "human_cpc_phonetic" / "best_model.pt"


@unittest.skipUnless(
    _HUMAN_CKPT.exists() and _HUMAN_CPC.exists(),
    "human VQ-VAE + CPC checkpoints not present",
)
class TestRunPipelineRealCheckpoints(unittest.TestCase):
    """Real tokenizer + auto-loaded calibrator, end-to-end on synthetic audio."""

    def test_full_chain_runs_and_is_schema_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(tmp, _bursts(periods=8))

            # Default: auto-loads checkpoints/nbd_calibrator/human.pkl.
            report, trace = run_pipeline([path], "human")
            self.assertEqual(validate_report(report), [])
            self.assertTrue(trace["schema_valid"])
            self.assertTrue(trace["tokenizer_available"])
            self.assertTrue(trace["calibrator_fitted"])

            # Force threshold fallback -> real VQ-VAE must emit tokens on onsets.
            report2, trace2 = run_pipeline([path], "human", calibrator=False)
            self.assertEqual(validate_report(report2), [])
            self.assertTrue(trace2["tokenizer_available"])
            self.assertGreater(trace2["n_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
