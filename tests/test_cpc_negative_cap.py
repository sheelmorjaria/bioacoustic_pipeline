"""Tests: CPCTrainer consumes TrainingConfig.negative_samples (§45 M8).

``negative_samples`` was defined but never passed to ``compute_loss`` —
stock training ran full-pool InfoNCE, whose gradient stalls at the
ln(B*T) random floor (the regime the compute_loss docstring itself
describes). These tests pin the wiring, plus one end-to-end epoch on a
labeled dataset with drop_labels=True (Arm A's no-leakage path).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401  (import gate; used indirectly by the suite)

    _TORCH = True
except ImportError:
    _TORCH = False

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel, CPCTrainer, TrainingConfig
from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation


@unittest.skipUnless(_TORCH, "needs torch")
class TestNegativeSamplesWiring(unittest.TestCase):
    def _tiny(self):
        return CPCModel(
            encoder_config=EncoderConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=8,
                num_channels=(4, 8, 8),
                kernel_sizes=(3, 3, 3),
            ),
            steps_ahead=2,
        )

    def test_train_epoch_passes_n_negatives(self):
        model = self._tiny()
        seen_kwargs: dict = {}

        class Recording(CPCTrainer):
            def train_epoch(self):
                # intercept the call by shadowing model.compute_loss args
                orig = self.model.compute_loss

                def wrapper(z, c, p, temperature=0.07, n_negatives=None):
                    seen_kwargs["n_negatives"] = n_negatives
                    return orig(z, c, p, temperature, n_negatives=n_negatives)

                self.model.compute_loss = wrapper  # type: ignore[method-assign]
                try:
                    return super().train_epoch()
                finally:
                    self.model.compute_loss = orig  # type: ignore[method-assign]

        with tempfile.TemporaryDirectory() as td:
            t = np.arange(int(0.4 * 8000)) / 8000
            sf.write(
                Path(td) / "a.wav", (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 8000
            )
            cfg = TrainingConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=8,
                sequence_length=8,
                steps_ahead=2,
                overlap=2,
                batch_size=2,
                num_epochs=1,
                checkpoint_dir=str(Path(td) / "ckpt"),
                negative_samples=7,
            )
            from bioacoustic_pipeline.embeddings.cpc_trainer import AudioSequenceDataset

            ds = AudioSequenceDataset(str(Path(td) / "a.wav"), cfg)
            trainer = Recording(model, cfg, train_dataset=ds)
            trainer.train_epoch()
        self.assertEqual(seen_kwargs["n_negatives"], 7)

    def test_stock_trainer_one_epoch_on_labeled_dataset_drop_labels(self):
        """Arm A end to end: labeled dataset, labels mechanically absent."""
        from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
            LabeledAudioSequenceDataset,
        )

        with tempfile.TemporaryDirectory() as td:
            t = np.arange(int(0.4 * 8000)) / 8000
            ann = []
            for c in (1, 2):
                name = f"{c}_0.wav"
                sig = 0.5 * np.sin(2 * np.pi * (150 + 100 * c) * t)
                sf.write(Path(td) / name, sig.astype(np.float32), 8000)
                ann.append(CallAnnotation(name, c, c, 1, 0, 0, 0, 0))
            cfg = TrainingConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=8,
                sequence_length=8,
                steps_ahead=2,
                overlap=2,
                batch_size=2,
                num_epochs=1,
                checkpoint_dir=str(Path(td) / "ckpt"),
                negative_samples=5,
            )
            ds = LabeledAudioSequenceDataset(td, ann, cfg, drop_labels=True)
            trainer = CPCTrainer(self._tiny(), cfg, train_dataset=ds)
            metrics = trainer.train_epoch()
            self.assertTrue(np.isfinite(metrics["train_loss"]))


if __name__ == "__main__":
    unittest.main()
