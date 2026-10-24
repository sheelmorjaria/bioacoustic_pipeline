"""Tests: caller-conditioned CPC (embeddings.caller_conditioned) — Arm B.

Mechanics only: shape contract, UNK(None)-path equivalence with the parent,
conditioning changes outputs, same-file masking zeroes the caller term, loss
decreases on synthetic caller-clustered audio (a smoke mechanic, not a
scientific threshold), trainer epoch runs finite, backbone round-trips a
checkpoint. Torch-gated; no network, no real checkpoints.

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

from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel, TrainingConfig
from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation


def _tiny_config(**kw):
    base = dict(
        sample_rate=8000,
        frame_size_ms=10,
        hidden_dim=8,
        sequence_length=8,
        steps_ahead=2,
        overlap=2,
        batch_size=4,
        num_epochs=1,
        checkpoint_dir="checkpoints/_never_used",
    )
    base.update(kw)
    return TrainingConfig(**base)


def _tiny_encoder():
    return EncoderConfig(
        sample_rate=8000,
        frame_size_ms=10,
        hidden_dim=8,
        num_channels=(4, 8, 8),
        kernel_sizes=(3, 3, 3),
    )


def _audio(batch=4, t=8, frame=80, callers=None, seed=0):
    """Synthetic frames; when callers given, per-caller base frequency plants
    a mechanically learnable caller signature."""
    out = np.zeros((batch, t, frame), dtype=np.float32)
    for b in range(batch):
        base = 0.0 if callers is None else 40.0 * int(callers[b])
        phase = 2 * np.pi * (100 + base) * np.arange(t * frame) / 8000.0
        out[b] = (0.5 * np.sin(phase)).reshape(t, frame)
    g = torch.Generator().manual_seed(seed)
    return torch.from_numpy(out) + 0.01 * torch.randn(batch, t, frame, generator=g)


@unittest.skipUnless(_TORCH, "needs torch")
class TestCallerConditionedCPC(unittest.TestCase):
    def _model(self, n_callers=3, steps_ahead=2):
        from bioacoustic_pipeline.embeddings.caller_conditioned import (
            CallerConditionedCPC,
        )

        torch.manual_seed(0)
        return CallerConditionedCPC(
            encoder_config=_tiny_encoder(), steps_ahead=steps_ahead, n_callers=n_callers
        )

    def test_shapes_match_parent_contract(self):
        model = self._model().eval()
        audio = _audio()
        z, ctx, preds = model(audio, caller_ids=torch.tensor([0, 1, 2, 0]))
        self.assertEqual(tuple(z.shape), (4, 8, 8))
        self.assertEqual(tuple(ctx.shape), (4, 8, 8))
        self.assertEqual(len(preds), 2)
        self.assertEqual(tuple(preds[0].shape), (4, 8, 8))

    def test_none_path_equals_parent_forward(self):
        model = self._model().eval()
        audio = _audio(seed=3)
        z1, c1, p1 = model(audio, caller_ids=None)
        z2, c2, p2 = CPCModel.forward(model, audio)
        self.assertTrue(torch.allclose(z1, z2, atol=1e-6))
        self.assertTrue(torch.allclose(c1, c2, atol=1e-6))
        self.assertTrue(torch.allclose(p1[0], p2[0], atol=1e-6))

    def test_conditioning_changes_context(self):
        model = self._model().eval()
        audio = _audio(seed=1)
        _, c_unk, _ = model(audio, caller_ids=None)
        _, c_cond, _ = model(audio, caller_ids=torch.tensor([0, 1, 2, 0]))
        self.assertGreater(float((c_unk - c_cond).abs().max()), 1e-6)

    def test_latents_stay_unconditioned(self):
        # the self-supervised InfoNCE pool must not see caller vectors
        model = self._model().eval()
        audio = _audio(seed=2)
        z_unk, _, _ = model(audio, caller_ids=None)
        z_cond, _, _ = model(audio, caller_ids=torch.tensor([1, 1, 1, 1]))
        self.assertTrue(torch.allclose(z_unk, z_cond, atol=0.0))

    def test_same_file_only_batch_masks_caller_term_to_zero(self):
        model = self._model()
        audio = _audio(seed=4)
        _, ctx, _ = model(audio, caller_ids=torch.tensor([0, 1, 2, 0]))
        loss = model.caller_contrastive_loss(
            ctx,
            caller_ids=torch.tensor([0, 1, 2, 0]),
            file_ids=torch.tensor([7, 7, 7, 7]),
        )
        self.assertEqual(float(loss), 0.0)

    def test_loss_decreases_on_caller_clustered_audio(self):
        model = self._model(n_callers=2)
        callers = [0, 1, 0, 1]
        audio = _audio(callers=callers, seed=5)
        ids = torch.tensor(callers)
        files = torch.tensor([0, 1, 2, 3])
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        losses = []
        for _ in range(12):
            z, ctx, preds = model(audio, caller_ids=ids)
            total = model.compute_loss(z, ctx, preds, 0.07) + 0.5 * (
                model.caller_contrastive_loss(ctx, ids, files, temperature=0.1)
            )
            opt.zero_grad()
            total.backward()
            opt.step()
            losses.append(float(total))
        self.assertLess(losses[-1], losses[0])


@unittest.skipUnless(_TORCH, "needs torch")
class TestCallerCPCTrainer(unittest.TestCase):
    def test_one_epoch_finite(self):
        from bioacoustic_pipeline.embeddings.caller_conditioned import (
            CallerCPCTrainer,
            CallerConditionedCPC,
        )
        from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
            LabeledAudioSequenceDataset,
        )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            t = np.arange(int(0.4 * 8000)) / 8000
            ann = []
            for c in (1, 2):
                for f in range(3):
                    name = f"{c}_{f}.wav"
                    sig = 0.5 * np.sin(2 * np.pi * (150 + 120 * c) * t + 0.3 * f)
                    sf.write(td / name, sig.astype(np.float32), 8000)
                    ann.append(CallAnnotation(name, c, c, 1, 0, 0, 0, 0))
            cfg = _tiny_config()
            ds = LabeledAudioSequenceDataset(td, ann, cfg)
            model = CallerConditionedCPC(encoder_config=_tiny_encoder(), steps_ahead=2, n_callers=3)
            trainer = CallerCPCTrainer(model, cfg, train_dataset=ds, n_callers=3)
            metrics = trainer.train_epoch()
            self.assertTrue(np.isfinite(metrics["train_loss"]))
            self.assertIn("caller_loss", metrics)


@unittest.skipUnless(_TORCH, "needs torch")
class TestCallerConditionedBackbone(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        from bioacoustic_pipeline.embeddings.caller_conditioned import (
            CallerConditionedBackbone,
            CallerConditionedCPC,
        )

        model = CallerConditionedCPC(encoder_config=_tiny_encoder(), steps_ahead=2, n_callers=3)
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best_model.pt"
            torch.save({"model_state_dict": model.state_dict(), "n_callers": 3}, ckpt)
            bb = CallerConditionedBackbone(
                name="bat_cpc_caller_selftest",
                provenance_name="bat_cpc_caller",
                checkpoint=ckpt,
            )
            self.assertTrue(bb.available, bb.load_error)
            vec = bb.embed(_audio(batch=1).numpy()[0, 0], 8000)
            self.assertEqual(vec.shape, (8,))
            self.assertEqual(vec.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
