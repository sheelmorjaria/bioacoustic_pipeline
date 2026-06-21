"""CPCBackbone device-consistency smoke (regression guard).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest
from pathlib import Path

import numpy as np

from bioacoustic_pipeline.embeddings.backbones import CPCBackbone

ROOT = Path(__file__).resolve().parents[1]
_CKPT = ROOT / "checkpoints" / "human_cpc_phonetic" / "best_model.pt"


@unittest.skipUnless(_CKPT.exists(), "needs checkpoints/human_cpc_phonetic/best_model.pt")
class TestCPCBackboneDevice(unittest.TestCase):
    """embed/embed_frames keep model + input on one device (no CUDA/CPU mismatch)."""

    def test_embed_frames_device_consistent(self):
        cpc = CPCBackbone()
        self.assertTrue(cpc.available, cpc.load_error)
        audio = np.random.randn(16000).astype(np.float32)  # 1 s @ 16 kHz
        frames = cpc.embed_frames(audio, 16000)
        self.assertEqual(frames.ndim, 2)
        self.assertEqual(frames.shape[1], cpc.embed_dim)
        self.assertEqual(frames.dtype, np.float32)

    def test_embed_pools_to_one_vector(self):
        cpc = CPCBackbone()
        if not cpc.available:
            self.skipTest("CPC checkpoint unavailable")
        audio = np.random.randn(16000).astype(np.float32)
        vec = cpc.embed(audio, 16000)
        self.assertEqual(vec.shape, (cpc.embed_dim,))
        self.assertEqual(vec.dtype, np.float32)

    def test_short_audio_returns_zero_pad(self):
        """<2 frames (10 ms hop) short-circuits to a (2, dim) zero pad, not a crash."""
        cpc = CPCBackbone()
        if not cpc.available:
            self.skipTest("CPC checkpoint unavailable")
        short = np.zeros(80, dtype=np.float32)  # 5 ms @ 16 kHz → 0 frames
        frames = cpc.embed_frames(short, 16000)
        self.assertEqual(frames.shape, (2, cpc.embed_dim))


if __name__ == "__main__":
    unittest.main()
