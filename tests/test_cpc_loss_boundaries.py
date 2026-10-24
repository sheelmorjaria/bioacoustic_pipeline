"""Tests: §46 loss boundaries — full-pool exclusion + valid-anchor reduction.

Gate A: the FULL-pool branch (n_negatives=None or >= pool) also masks the
anchor's exact positive (the sampled branch got it first; this closes the
latent legacy path).
Gate B: masked_cross_entropy divides by VALID anchors only — an anchor whose
negatives are all -inf contributes no loss and no denominator mass, instead
of diluting the mean (in per-frame eval, a diluted zero would read as a
falsely 'perfect' prediction).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


@unittest.skipUnless(_TORCH, "needs torch")
class TestMaskedCrossEntropy(unittest.TestCase):
    def test_valid_anchor_reduction_excludes_zero_loss_anchors(self):
        from bioacoustic_pipeline.embeddings.loss_utils import masked_cross_entropy

        logits = torch.tensor(
            [[2.0, 1.0], [2.0, float("-inf")]]
        )  # anchor 1: no eligible negatives
        labels = torch.zeros(2, dtype=torch.long)
        loss = masked_cross_entropy(logits, labels)
        # standard mean would be ~(0.313 + 0)/2 = 0.157; valid-anchor = 0.313
        self.assertGreater(float(loss), 0.2, "loss diluted by invalid anchor")
        ref = torch.nn.functional.cross_entropy(
            logits[:1], labels[:1]
        )
        self.assertAlmostEqual(float(loss), float(ref), places=5)

    def test_all_valid_behaves_like_standard_mean(self):
        from bioacoustic_pipeline.embeddings.loss_utils import masked_cross_entropy

        g = torch.Generator().manual_seed(0)
        logits = torch.randn(16, 6, generator=g)
        labels = torch.zeros(16, dtype=torch.long)
        a = masked_cross_entropy(logits, labels)
        b = torch.nn.functional.cross_entropy(logits, labels)
        self.assertAlmostEqual(float(a), float(b), places=5)

    def test_no_valid_anchor_returns_zero_not_nan(self):
        from bioacoustic_pipeline.embeddings.loss_utils import masked_cross_entropy

        logits = torch.tensor([[1.0, float("-inf")], [2.0, float("-inf")]])
        labels = torch.zeros(2, dtype=torch.long)
        loss = masked_cross_entropy(logits, labels)
        self.assertEqual(float(loss), 0.0)


@unittest.skipUnless(_TORCH, "needs torch")
class TestFullPoolPositiveExclusion(unittest.TestCase):
    def test_full_pool_masks_exact_positive(self):
        from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
        from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel

        torch.manual_seed(0)
        m = CPCModel(
            encoder_config=EncoderConfig(
                sample_rate=8000, frame_size_ms=10, hidden_dim=8,
                num_channels=(4, 8, 8), kernel_sizes=(3, 3, 3),
            ),
            steps_ahead=1,
        ).eval()
        B, T, H = 1, 4, 8
        z = torch.randn(B, T, H)
        ctx = torch.randn(B, T, H)
        loss = m.compute_loss(z, ctx, [None], 0.07, n_negatives=None)  # full pool
        # hand reference: full-pool contrast with the positive column dropped
        import torch.nn.functional as F

        c = F.normalize(ctx[:, : T - 1, :], dim=2).reshape(-1, H)
        pos = F.normalize(z[:, 1:, :], dim=2).reshape(-1, H)
        pool = F.normalize(z.reshape(-1, H), dim=1)
        pos_logit = (c * pos).sum(1, keepdim=True) / 0.07
        neg_logits = (c @ pool.T) / 0.07
        pos_pool_idx = torch.arange(1, T)
        neg_logits = neg_logits.masked_fill(
            torch.arange(pool.shape[0]).unsqueeze(0) == pos_pool_idx.unsqueeze(1),
            float("-inf"),
        )
        from bioacoustic_pipeline.embeddings.loss_utils import masked_cross_entropy

        ref = masked_cross_entropy(
            torch.cat([pos_logit, neg_logits], dim=1), torch.zeros(T - 1, dtype=torch.long)
        )
        self.assertAlmostEqual(float(loss), float(ref), places=5)


if __name__ == "__main__":
    unittest.main()
