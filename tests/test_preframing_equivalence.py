"""Tests: preload equivalence — the RAM pre-framing path returns exactly the
data the on-the-fly path returns (§46 Phase 2.7 regression gate).

The 2-min/epoch speedup must not alter the data contract: same windows, same
labels, same source attribution, boundary window included.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.embeddings.cpc_trainer import TrainingConfig
from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation


@unittest.skipUnless(_TORCH, "needs torch")
class TestPreFramingEquivalence(unittest.TestCase):
    def _corpus(self, td: Path):
        sr = 8000
        t = np.arange(int(0.6 * sr)) / sr
        sig = 0.5 * np.sin(2 * np.pi * 400 * t).astype(np.float32)
        ann = []
        for c in (1, 2):
            name = f"audio_16k/f{c}.wav"
            (td / "audio_16k").mkdir(exist_ok=True)
            sf.write(td / name, sig * (0.5 + 0.1 * c), sr)
            ann.append(CallAnnotation(name, c, 0, 11, 0, 0, 0, 0, start_ms=50.0, end_ms=550.0))
        return ann

    def _ds(self, td, preload):
        from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
            LabeledAudioSequenceDataset,
        )

        cfg = TrainingConfig(
            sample_rate=8000,
            frame_size_ms=10,
            hidden_dim=8,
            sequence_length=16,
            steps_ahead=2,
            overlap=2,
            device="cpu",
        )
        return LabeledAudioSequenceDataset(td, self._corpus(td), cfg, preload=preload)

    def test_interior_and_boundary_windows_identical(self):
        with tempfile.TemporaryDirectory() as td0, tempfile.TemporaryDirectory() as td1:
            ref = self._ds(Path(td0), preload=False)
            pre = self._ds(Path(td1), preload=True)
            self.assertEqual(len(ref), len(pre))
            for idx in (0, len(ref) // 2, len(ref) - 1):  # interior + boundary
                r, p = ref[idx], pre[idx]
                torch.testing.assert_close(r[0], p[0])  # context
                torch.testing.assert_close(r[1], p[1])  # future
                self.assertEqual((r[2], r[3], r[4]), (p[2], p[3], p[4]))  # attribution

    def test_file_names_and_lengths_agree(self):
        with tempfile.TemporaryDirectory() as td0, tempfile.TemporaryDirectory() as td1:
            ref = self._ds(Path(td0), preload=False)
            pre = self._ds(Path(td1), preload=True)
            self.assertEqual(ref.file_names, pre.file_names)
            self.assertEqual(len(ref), len(pre))


if __name__ == "__main__":
    unittest.main()
