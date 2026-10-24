"""Tests: §46 training gates — positive exclusion + UNK head co-training.

Gate 1 (parent fix): the sampled-negative path of CPCModel.compute_loss must
mask negative columns that ARE the anchor's exact positive (sampled pools
collide with it). §45 caveat: both arms trained pre-fix, so their paired
comparison remains internally fair; every future run gets the fix.

Gate 2: the UNK population head must co-train on known-caller batches —
held-out recipients route to it for their own-baseline term in
cross_perplexity, so an untrained UNK head would corrupt ΔL for exactly the
callers the Track-C protocol cares about.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest
from unittest import mock

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


@unittest.skipUnless(_TORCH, "needs torch")
class TestPositiveExclusionGate(unittest.TestCase):
    def test_collision_mask_helper(self):
        from bioacoustic_pipeline.embeddings.cpc_trainer import (
            positive_collision_mask,
        )

        pos = torch.tensor([2, 7, 0])
        negs = torch.tensor([2, 0, 5, 7])
        mask = positive_collision_mask(pos, negs)
        self.assertEqual(tuple(mask.shape), (3, 4))
        self.assertEqual(mask.tolist()[0], [True, False, False, False])
        self.assertEqual(mask.tolist()[1], [False, False, False, True])
        self.assertEqual(mask.tolist()[2], [False, True, False, False])

    def test_compute_loss_masks_forced_collision(self):
        """randint forced to draw an anchor's positive: loss must match the
        hand-computed reference with that column excluded (no label leak)."""
        from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
        from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel

        torch.manual_seed(0)
        m = CPCModel(
            encoder_config=EncoderConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=8,
                num_channels=(4, 8, 8),
                kernel_sizes=(3, 3, 3),
            ),
            steps_ahead=1,
        ).eval()
        B, T, H = 1, 4, 8
        z = torch.randn(B, T, H)
        ctx = torch.randn(B, T, H)
        neg_idx = torch.tensor([2, 3, 0])  # index 2 IS the k=1 positive of anchor 1
        with mock.patch(
            "bioacoustic_pipeline.embeddings.cpc_trainer.torch.randint", return_value=neg_idx
        ):
            loss = m.compute_loss(z, ctx, [None], 0.07, n_negatives=3)
        # reference: identical computation with the colliding column dropped
        import torch.nn.functional as F

        c = F.normalize(ctx[:, : T - 1, :], dim=2).reshape(-1, H)
        pos = F.normalize(z[:, 1:, :], dim=2).reshape(-1, H)
        pool = F.normalize(z.reshape(-1, H), dim=1)
        negs = pool[neg_idx]
        pos_logit = (c * pos).sum(1, keepdim=True) / 0.07
        neg_logits = (c @ negs.T) / 0.07
        pos_pool_idx = torch.arange(1, T)
        coll = neg_idx.unsqueeze(0) == pos_pool_idx.unsqueeze(1)
        neg_logits = neg_logits.masked_fill(coll, float("-inf"))
        ref = F.cross_entropy(
            torch.cat([pos_logit, neg_logits], dim=1),
            torch.zeros(T - 1, dtype=torch.long),
        )
        self.assertAlmostEqual(float(loss), float(ref), places=5)


@unittest.skipUnless(_TORCH, "needs torch")
class TestUnkHeadCoTrainingGate(unittest.TestCase):
    def _model(self):
        from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
        from bioacoustic_pipeline.embeddings.multi_head_cpc import (
            SharedEncoderMultiHeadCPC,
        )

        torch.manual_seed(0)
        return SharedEncoderMultiHeadCPC(
            encoder_config=EncoderConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=16,
                num_channels=(8, 16, 16),
                kernel_sizes=(3, 3, 3),
            ),
            steps_ahead=2,
            n_callers=3,
        ).train()

    def _audio(self, seed=0, batch=4):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(batch, 12, 80, generator=g)

    def test_unk_head_receives_gradients_on_known_caller_batch(self):
        m = self._model()
        ids = torch.tensor([0, 1, 2, 0])
        files = torch.arange(4)
        loss = m.head_routing_loss(self._audio(), ids, files, temperature=0.1)
        loss.backward()
        for k in range(2):
            grad = m.heads["unk"][k].weight.grad
            self.assertIsNotNone(grad, f"UNK head step {k} got no gradient")
            self.assertTrue(bool((grad != 0).any()), f"UNK head step {k} grad all zero")

    def test_unk_head_updates_with_optimizer_step(self):
        m = self._model()
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        before = m.heads["unk"][0].weight.detach().clone()
        ids = torch.tensor([1, 1, 2, 2])
        files = torch.arange(4)
        loss = m.head_routing_loss(self._audio(seed=1), ids, files, temperature=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        after = m.heads["unk"][0].weight.detach().clone()
        self.assertFalse(torch.equal(before, after), "UNK head did not update")

    def test_caller_heads_still_isolated(self):
        m = self._model()
        ids = torch.tensor([0, 0, 0, 0])  # only caller 0 present
        files = torch.arange(4)
        loss = m.head_routing_loss(self._audio(seed=2), ids, files, temperature=0.1)
        loss.backward()
        self.assertIsNone(m.heads["1"][0].weight.grad)
        self.assertIsNone(m.heads["2"][0].weight.grad)
        self.assertIsNotNone(m.heads["0"][0].weight.grad)


if __name__ == "__main__":
    unittest.main()
