"""Tests: XPU smoke — CPC forward, optimizer step, data batch on the Intel GPU.

Gated on ``torch.xpu.is_available()`` so CPU-only environments and CI skip
cleanly. Exercises exactly the §45 training path components on-device:
CPCModel forward + backward + AdamW step, LabeledAudioSequenceDataset batch
transfer, and the augment transform on the device stream. Plus an optional
ipex.optimize smoke when intel-extension-for-pytorch is importable.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import torch

    _TORCH = True
    _XPU = torch.xpu.is_available()
except ImportError:
    _TORCH = _XPU = False


@unittest.skipUnless(_TORCH and _XPU, "needs torch with a visible XPU device")
class TestXPUSmoke(unittest.TestCase):
    def test_cpc_forward_backward_step_on_xpu(self):
        from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
        from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel

        torch.manual_seed(0)
        model = CPCModel(
            encoder_config=EncoderConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=16,
                num_channels=(8, 16, 16),
                kernel_sizes=(3, 3, 3),
            ),
            steps_ahead=2,
        ).to("xpu")
        audio = torch.randn(4, 16, 80, device="xpu")
        z, ctx, preds = model(audio)
        self.assertEqual(z.device.type, "xpu")
        loss = model.compute_loss(z, ctx, preds, 0.07, n_negatives=8)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        self.assertTrue(torch.isfinite(loss).item())

    def test_labeled_batch_transfers_and_trains_on_xpu(self):
        import numpy as np
        import soundfile as sf

        from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
        from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel, TrainingConfig
        from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
            LabeledAudioSequenceDataset,
        )
        from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as td:
            t = np.arange(int(0.4 * 8000)) / 8000
            ann = []
            for c in (1, 2):
                name = f"{c}_0.wav"
                sf.write(
                    Path(td) / name,
                    (0.5 * np.sin(2 * np.pi * (150 + 100 * c) * t)).astype(np.float32),
                    8000,
                )
                ann.append(CallAnnotation(name, c, c, 1, 0, 0, 0, 0))
            cfg = TrainingConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=16,
                sequence_length=8,
                steps_ahead=2,
                overlap=2,
                batch_size=2,
                num_epochs=1,
                checkpoint_dir=str(Path(td) / "ckpt"),
                device="xpu",
            )
            ds = LabeledAudioSequenceDataset(td, ann, cfg)
            model = CPCModel(
                encoder_config=EncoderConfig(
                    sample_rate=8000,
                    frame_size_ms=10,
                    hidden_dim=16,
                    num_channels=(8, 16, 16),
                    kernel_sizes=(3, 3, 3),
                ),
                steps_ahead=2,
            ).to("xpu")
            ctx, fut, caller, code, fid = ds[0]
            batch = (
                ctx.unsqueeze(0).to("xpu"),
                fut.unsqueeze(0).to("xpu"),
                torch.tensor([caller]),
                torch.tensor([code]),
                torch.tensor([fid]),
            )
            audio = torch.cat([batch[0], batch[1]], dim=1)
            z, c, p = (
                model(audio, caller_ids=None)
                if hasattr(model, "caller_embedding")
                else model(audio)
            )
            loss = model.compute_loss(z, c, p, 0.07, n_negatives=5)
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())

    def test_augment_on_device_stream(self):
        from bioacoustic_pipeline.embeddings.augment import add_white_noise

        x = torch.ones(1000, dtype=torch.float32)
        out = add_white_noise(x, snr_db=10.0, rng=__import__("numpy").random.RandomState(0))
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.from_numpy(out[:10]).to("xpu").isfinite().all().item())

    def test_ipex_optimize_smoke(self):
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
        except ImportError:
            self.skipTest("intel-extension-for-pytorch not installed")
        from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
        from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel

        model = CPCModel(
            encoder_config=EncoderConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=16,
                num_channels=(8, 16, 16),
                kernel_sizes=(3, 3, 3),
            ),
            steps_ahead=2,
        ).to("xpu")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model, opt = ipex.optimize(model, optimizer=opt)
        z, c, p = model(torch.randn(2, 8, 80, device="xpu"))
        self.assertEqual(z.device.type, "xpu")


if __name__ == "__main__":
    unittest.main()
