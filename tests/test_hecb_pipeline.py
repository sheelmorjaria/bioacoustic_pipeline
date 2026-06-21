"""Tests: HECB segmentation in run_pipeline + DualStreamSegment sub-call threading.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.orchestration import run_pipeline
from bioacoustic_pipeline.representation.dual_stream import (
    build_dual_stream,
    build_dual_stream_from_calls,
)
from bioacoustic_pipeline.segmentation.hierarchical import CallEvent, InternalSegment
from bioacoustic_pipeline.segmentation import hecb as hecb_mod
from bioacoustic_pipeline.verdict_engine import is_valid, validate_report

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
_HECB_CKPT = ROOT / "checkpoints" / "hecb" / "crnn.pt"


def _tone(freq, dur, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _bursts(periods=5, period=0.5, freq=440.0, sr=SR):
    tone = _tone(freq, period / 2, sr)
    silence = np.zeros(int(sr * period / 2), dtype=np.float32)
    return np.concatenate([np.concatenate([tone, silence]) for _ in range(periods)])


def _write_wav(tmp, audio, sr=SR):
    path = str(Path(tmp) / "bursts.wav")
    sf.write(path, audio.astype(np.float32), sr)
    return path


class _FakeTokenizer:
    def __init__(self):
        self.provenance_name = "human_vqvae_large"
        self._n = 0

    @property
    def available(self):
        return True

    @property
    def load_error(self):
        return None

    def tokenize(self, audio, sr):  # noqa: ARG002
        self._n += 1
        return self._n % 5


class TestDualStreamSubcallThreading(unittest.TestCase):
    """InternalSegment → DualStreamSegment.sub_segments (Part A)."""

    def test_build_dual_stream_from_calls_threads_subcalls(self):
        sr = 8000
        audio = np.ones(int(0.5 * sr), dtype=np.float32)
        calls = [
            CallEvent(
                call_id="call_0",
                onset_s=0.0,
                offset_s=0.4,
                crnn_confidence=0.9,
                internal_segments=[
                    InternalSegment(parent_call_id="call_0", start_s=0.1, end_s=0.2),
                    InternalSegment(parent_call_id="call_0", start_s=0.2, end_s=0.3),
                ],
            ),
            CallEvent(call_id="call_1", onset_s=0.4, offset_s=0.48, crnn_confidence=0.7),
        ]
        fp_fn = lambda a, s: np.array([float(a.mean()), float(a.std())], dtype=np.float32)  # noqa: E731
        ds = build_dual_stream_from_calls(calls, audio, sr, fingerprint_fn=fp_fn)
        self.assertEqual(len(ds), 2)
        # parent 0 has 2 sub-segments; parent 1 has none
        self.assertEqual(ds[0].n_sub_segments, 2)
        self.assertEqual(ds[1].n_sub_segments, 0)
        # children link back via parent_call_id
        for child in ds[0].sub_segments:
            self.assertEqual(child.parent_call_id, "call_0")
            self.assertEqual(child.boundary_type, "hecb_internal")
        # parent is the whole call
        self.assertEqual(ds[0].boundary_type, "hecb_call")
        # as_event carries the hierarchy
        ev = ds[0].as_event()
        self.assertEqual(ev["n_sub"], 2)
        self.assertEqual(ev["parent_call_id"], None)  # parent has no parent

    def test_build_dual_stream_flat_remains_backward_compatible(self):
        """Flat NBD build still works + produces no sub_segments."""
        sr = 8000

        class _Seg:
            def __init__(self, a, b):
                self.start_ms, self.end_ms = a, b
                self.boundary_type, self.confidence = "hard", 1.0
                self.audio, self.sr = np.ones(800, dtype=np.float32), sr

        ds = build_dual_stream([_Seg(0, 100), _Seg(100, 200)])
        self.assertEqual(len(ds), 2)
        for d in ds:
            self.assertEqual(d.n_sub_segments, 0)
            self.assertIsNone(d.parent_call_id)


class TestRunPipelineSegmentation(unittest.TestCase):
    """segmentation='hecb' / 'nbd' dispatch + trace (Part C)."""

    def test_default_segmentation_is_nbd(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(tmp, _bursts())
            report, trace = run_pipeline(
                [path], "human", tokenizer=_FakeTokenizer(), calibrator=False
            )
        self.assertTrue(is_valid(report), validate_report(report))
        self.assertEqual(trace["segmentation"], "nbd")
        self.assertIsNone(trace["hecb"])

    def test_hecb_falls_back_when_no_checkpoint(self):
        """human has no HECB checkpoint → falls back to NBD, trace flags unavailable."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_wav(tmp, _bursts())
            report, trace = run_pipeline(
                [path], "human", tokenizer=_FakeTokenizer(), segmentation="hecb", calibrator=False
            )
        self.assertTrue(is_valid(report), validate_report(report))
        self.assertEqual(trace["segmentation"], "hecb")
        self.assertIsNotNone(trace["hecb"])
        self.assertFalse(trace["hecb"]["available"])

    @unittest.skipUnless(
        _HECB_CKPT.exists() and hecb_mod.available,
        "needs checkpoints/hecb/crnn.pt + torch",
    )
    def test_hecb_meerkat_runs_when_checkpoint_present(self):
        """meerkat has an HECB checkpoint → CRNN runs, trace flags available."""
        with tempfile.TemporaryDirectory() as tmp:
            # meerkat profile is 8 kHz; write bursts at 8 kHz
            path = str(Path(tmp) / "mk.wav")
            sf.write(path, _bursts(sr=8000).astype(np.float32), 8000)
            report, trace = run_pipeline(
                [path], "meerkat", tokenizer=_FakeTokenizer(), segmentation="hecb"
            )
        self.assertTrue(is_valid(report), validate_report(report))
        self.assertEqual(trace["segmentation"], "hecb")
        self.assertTrue(trace["hecb"]["available"])
        # n_internal_boundaries is a non-negative int (CRNN may find 0 calls on synthetic)
        self.assertIsInstance(trace["hecb"]["n_internal_boundaries"], int)


if __name__ == "__main__":
    unittest.main()
