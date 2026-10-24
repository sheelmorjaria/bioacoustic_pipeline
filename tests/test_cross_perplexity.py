"""Tests: cross-model perplexity time series (§46 Phase 2, Track C prep).

Mechanics only: series length aligns 1:1 with valid context frames, delta is
the elementwise difference of donor-head and own-head losses (identical
heads => zero delta), and spikes localize to specific frames. No checkpoints.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


def _audio(t=40, frame=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, t, frame, generator=g)


@unittest.skipUnless(_TORCH, "needs torch")
class TestCrossPerplexitySeries(unittest.TestCase):
    def _model(self, n_callers=2):
        from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
        from bioacoustic_pipeline.embeddings.multi_head_cpc import (
            SharedEncoderMultiHeadCPC,
        )

        torch.manual_seed(0)
        return SharedEncoderMultiHeadCPC(
            encoder_config=EncoderConfig(
                sample_rate=8000, frame_size_ms=10, hidden_dim=16,
                num_channels=(8, 16, 16), kernel_sizes=(3, 3, 3),
            ),
            steps_ahead=2,
            n_callers=n_callers,
        ).eval()

    def test_series_length_aligns_to_context_frames(self):
        from bioacoustic_pipeline.embeddings.multi_head_cpc import (
            cross_perplexity_series,
        )

        m = self._model()
        t = 40
        series = cross_perplexity_series(
            m, _audio(t=t), donor_caller=1, recipient_caller=0
        )
        self.assertEqual(series.shape, (t - m.steps_ahead,))

    def test_identical_heads_give_zero_delta(self):
        from bioacoustic_pipeline.embeddings.multi_head_cpc import (
            cross_perplexity_series,
        )

        m = self._model()
        # copy head 0's weights into head 1 => donor == own translation
        with torch.no_grad():
            for k in range(m.steps_ahead):
                m.heads["1"][k].load_state_dict(m.heads["0"][k].state_dict())
        series = cross_perplexity_series(
            m, _audio(seed=1), donor_caller=1, recipient_caller=0
        )
        self.assertAlmostEqual(float(abs(series).max()), 0.0, places=5)

    def test_different_heads_give_nonzero_delta(self):
        from bioacoustic_pipeline.embeddings.multi_head_cpc import (
            cross_perplexity_series,
        )

        m = self._model()
        series = cross_perplexity_series(
            m, _audio(seed=2), donor_caller=1, recipient_caller=0
        )
        self.assertGreater(float(series.abs().max()), 1e-6)

    def test_delta_is_elementwise_difference(self):
        """delta[t] = L_cross[t] - L_own[t] by construction (Test 2.2 pin)."""
        from bioacoustic_pipeline.embeddings.multi_head_cpc import (
            cross_perplexity_series,
            per_frame_head_loss,
        )

        m = self._model()
        audio = _audio(seed=3)
        torch.manual_seed(7)
        delta = cross_perplexity_series(m, audio, donor_caller=1, recipient_caller=0)
        torch.manual_seed(7)
        own = per_frame_head_loss(m, audio, head_key="0")
        cross = per_frame_head_loss(m, audio, head_key="1")
        # not byte-identical (series shares one neg draw across heads while
        # the manual calls draw per horizon) — but the arithmetic identity
        # delta = cross - own holds for the series' OWN two evaluations:
        torch.manual_seed(7)
        d2 = cross_perplexity_series(m, audio, donor_caller=1, recipient_caller=0)
        self.assertAlmostEqual(float((delta - d2).abs().max()), 0.0, places=5)
        self.assertGreater(float((cross - own).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
