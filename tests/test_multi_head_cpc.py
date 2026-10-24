"""Tests: shared-encoder multi-head CPC (§46 Track-C instrument).

Architecture: encoder + AR context are SHARED and unconditioned; caller
specificity lives ONLY in per-caller linear prediction heads routed by
caller id. This keeps the "grammar" (context) population-level and makes the
"translation" (context -> future) caller-specific — the property the
cross-model perplexity analysis needs (donor head, recipient trajectory).

CRITICAL mechanics pinned here: under the PARENT CPCModel loss the heads
receive NO gradient (queries are the raw context — the documented §45
quirk), so this class carries its own head-routing loss; Test 1.4 asserts
the SELECTED head trains while unselected heads stay untouched — a test the
parent loss would fail for BOTH heads.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


def _config():
    from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig

    return EncoderConfig(
        sample_rate=8000,
        frame_size_ms=10,
        hidden_dim=16,
        num_channels=(8, 16, 16),
        kernel_sizes=(3, 3, 3),
    )


def _audio(batch=4, t=12, frame=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, t, frame, generator=g)


@unittest.skipUnless(_TORCH, "needs torch")
class TestSharedEncoderMultiHeadCPC(unittest.TestCase):
    def _model(self, n_callers=3):
        from bioacoustic_pipeline.embeddings.multi_head_cpc import (
            SharedEncoderMultiHeadCPC,
        )

        torch.manual_seed(0)
        return SharedEncoderMultiHeadCPC(
            encoder_config=_config(), steps_ahead=2, n_callers=n_callers
        ).eval()

    def test_forward_shape_matches_parent(self):
        from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel

        m = self._model()
        audio = _audio()
        z, ctx, preds = m(audio, caller_ids=torch.tensor([0, 1, 2, 0]))
        z2, c2, p2 = CPCModel.forward(m, audio)
        self.assertEqual(z.shape, z2.shape)
        self.assertEqual(ctx.shape, c2.shape)
        self.assertEqual(len(preds), len(p2))

    def test_shared_path_identical_heads_differ(self):
        m = self._model()
        audio = _audio(seed=1)
        ids_a = torch.tensor([0, 0, 0, 0])
        ids_b = torch.tensor([1, 1, 1, 1])
        z_a, c_a, p_a = m(audio, caller_ids=ids_a)
        z_b, c_b, p_b = m(audio, caller_ids=ids_b)
        self.assertTrue(torch.allclose(z_a, z_b, atol=0.0))  # shared encoder
        self.assertTrue(torch.allclose(c_a, c_b, atol=0.0))  # shared AR grammar
        self.assertFalse(torch.allclose(p_a[0], p_b[0], atol=1e-7))  # heads differ

    def test_unk_falls_back_to_population_head(self):
        m = self._model()
        audio = _audio(seed=2)
        _, _, p_unk = m(audio, caller_ids=None)
        _, _, p_zero = m(audio, caller_ids=torch.tensor([9, 9, 9, 9]))  # unseen
        self.assertTrue(torch.allclose(p_unk[0], p_zero[0], atol=0.0))

    def test_gradient_flow_isolation(self):
        """Selected head trains; unselected head untouched; encoder trains.

        This is the test the PARENT loss fails for both heads (heads get no
        gradient there) — the reason this class carries its own loss.
        """
        m = self._model().train()
        audio = _audio(seed=3)
        ids = torch.tensor([0, 0, 1, 1])  # two callers, both routed
        files = torch.tensor([0, 1, 2, 3])
        loss = m.head_routing_loss(audio, ids, files, temperature=0.1)
        loss.backward()
        for k in range(2):
            self.assertIsNotNone(m.heads["0"][k].weight.grad)  # used -> grad
            self.assertIsNotNone(m.heads["1"][k].weight.grad)
        self.assertIsNone(m.heads["2"][0].weight.grad)  # unused -> untouched
        self.assertIsNotNone(m.encoder.encoder[0].weight.grad)  # shared path trains

    def test_head_routing_loss_finite_and_shapes(self):
        m = self._model().train()
        audio = _audio(seed=4)
        ids = torch.tensor([0, 1, 2, 0])
        files = torch.arange(4)
        loss = m.head_routing_loss(audio, ids, files, temperature=0.1, n_negatives=5)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
