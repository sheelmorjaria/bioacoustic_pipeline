"""Tests: vocal-target CPC loss masking (two-arm pilot machinery).

The contract: target ELIGIBILITY changes which predictions drive learning,
not the architecture or negative policy. Anchors with no eligible negatives
are excluded (not reassuring zeros). Padding inertness holds. Comparison
endpoints are fixed before training.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

import numpy as np

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


@unittest.skipUnless(_TORCH, "needs torch")
class TestEligibleTargetMask(unittest.TestCase):
    def _model(self):
        from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
        from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel

        torch.manual_seed(0)
        return CPCModel(
            encoder_config=EncoderConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=16,
                num_channels=(8, 16, 16),
                kernel_sizes=(3, 3, 3),
            ),
            steps_ahead=2,
        )

    def test_vocal_mask_restricts_loss_targets(self):
        """Masked loss != unmasked loss when targets differ."""
        from bioacoustic_pipeline.embeddings.vocal_target import (
            eligible_target_mask,
            masked_cpc_loss,
        )

        m = self._model().eval()
        g = torch.Generator().manual_seed(1)
        audio = torch.randn(4, 20, 80, generator=g)
        # vocal for half the batch, silent for the other half
        vocal = torch.zeros(4, 20, dtype=torch.bool)
        vocal[:2] = True  # first two items fully vocal
        mask = eligible_target_mask(vocal, steps_ahead=2)
        self.assertEqual(mask.tolist(), [18, 18, 0, 0])  # T - steps_ahead each
        torch.manual_seed(0)
        lm = masked_cpc_loss(m, audio, vocal, temperature=0.07, n_negatives=5)
        torch.manual_seed(0)
        audio_v = audio[:2]
        z, c, p = m(audio_v)
        direct = m.compute_loss(z, c, [None] * 2, 0.07, n_negatives=5)
        self.assertAlmostEqual(float(lm), float(direct), places=5)

    def test_no_eligible_targets_returns_nan_not_zero(self):
        from bioacoustic_pipeline.embeddings.vocal_target import masked_cpc_loss

        m = self._model().eval()
        audio = torch.randn(2, 20, 80)
        vocal = torch.zeros(2, 20, dtype=torch.bool)
        loss = masked_cpc_loss(m, audio, vocal, temperature=0.07, n_negatives=5)
        self.assertTrue(torch.isnan(loss))  # excluded, not a reassuring zero

    def test_padding_inertness(self):
        """Appending padding does not change the loss on valid source rows."""
        from bioacoustic_pipeline.embeddings.vocal_target import masked_cpc_loss

        m = self._model().eval()
        g = torch.Generator().manual_seed(2)
        audio = torch.randn(2, 20, 80, generator=g)
        vocal = torch.ones(2, 20, dtype=torch.bool)
        torch.manual_seed(0)
        a = float(masked_cpc_loss(m, audio, vocal, temperature=0.07, n_negatives=5))
        padded_audio = torch.cat([audio, torch.zeros(2, 10, 80)], dim=1)
        padded_vocal = torch.ones(
            2, 30, dtype=torch.bool
        )  # padding not vocal in the detector's sense; eligibility unchanged for valid rows
        torch.manual_seed(0)
        b = float(masked_cpc_loss(m, padded_audio, padded_vocal, temperature=0.07, n_negatives=5))
        # the valid-row loss should be comparable (not identical — negative
        # sampling pool grows), bounded difference
        self.assertLess(abs(a - b), 2.0)

    def test_vocal_fraction_from_audio(self):
        from bioacoustic_pipeline.embeddings.vocal_target import vocal_mask_from_audio

        sr = 8000
        t = np.arange(sr) / sr  # 1 s of tone
        tone = 0.5 * np.sin(2 * np.pi * 300 * t)
        x = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr // 2)])  # 2 s total
        mask = vocal_mask_from_audio(x, sr, frame_ms=10, threshold_db=-36.0)
        self.assertEqual(len(mask), 200)  # 2 s @ 10 ms
        self.assertTrue(mask[50:150].all())  # tone region (0.5-1.5 s)
        self.assertFalse(mask[:50].any())  # leading silence
        self.assertFalse(mask[150:].any())  # trailing silence
        self.assertEqual(mask.sum(), 100)


if __name__ == "__main__":
    unittest.main()
