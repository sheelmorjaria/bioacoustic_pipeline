"""Tests: labeled multi-file window dataset (embeddings.labeled_sequence_dataset).

Torch + soundfile gated. Synthetic bat-rate WAVs via tempfile; mechanics only:
window arithmetic, label alignment (a window's labels are its FILE's labels),
split filtering, drop_labels arity, transform-before-cut. No checkpoints.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False

import numpy as np
import soundfile as sf

from bioacoustic_pipeline.embeddings.cpc_trainer import TrainingConfig
from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation

SR = 8000  # small sr keeps fixtures fast; arithmetic is sr-agnostic


def _config(**kw):
    base = dict(
        sample_rate=SR,
        frame_size_ms=10,  # 80 samples/frame
        sequence_length=16,
        steps_ahead=4,
        overlap=4,
    )
    base.update(kw)
    return TrainingConfig(**base)


def _write_bouts(td: Path, n_callers=3, files_per_caller=2, seconds=0.5):
    """Bat-schema annotations + FM-sweep wavs; returns (dir, annotations)."""
    t = np.arange(int(seconds * SR)) / SR
    ann = []
    for c in range(1, n_callers + 1):
        for f in range(files_per_caller):
            name = f"{c:02d}_{f}.wav"
            sweep = np.sin(2 * np.pi * (200 + 100 * c + 400 * t) * t).astype(np.float32)
            sf.write(td / name, sweep, SR)
            ann.append(
                CallAnnotation(
                    file_name=name,
                    emitter=c,
                    addressee=c + 1,
                    context=1 + (f % 3),
                    emit_pre=0,
                    addr_pre=0,
                    emit_post=0,
                    addr_post=0,
                )
            )
    return ann


@unittest.skipUnless(_TORCH, "needs torch")
class TestLabeledAudioSequenceDataset(unittest.TestCase):
    def _dataset(self, td, ann, config=None, **kw):
        from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
            LabeledAudioSequenceDataset,
        )

        return LabeledAudioSequenceDataset(td, ann, config or _config(), **kw)

    def test_window_shapes_and_count(self):
        # 0.5 s @ 10 ms = 50 frames; seq_len = 16+4 = 20, stride = 12
        # per file: (50-20)//12 + 1 = 3 windows; 6 files -> 18
        with tempfile.TemporaryDirectory() as td:
            ann = _write_bouts(Path(td))
            ds = self._dataset(Path(td), ann)
            self.assertEqual(len(ds), 18)
            ctx, fut, caller, context_code, file_id = ds[0]
            self.assertEqual(tuple(ctx.shape), (16, 80))
            self.assertEqual(tuple(fut.shape), (4, 80))
            self.assertEqual(ctx.dtype, torch.float32)

    def test_labels_align_to_window_file(self):
        with tempfile.TemporaryDirectory() as td:
            ann = _write_bouts(Path(td))
            ds = self._dataset(Path(td), ann)
            by_file = {a.file_name: a for a in ann}
            names = ds.file_names
            for idx in range(len(ds)):
                _, _, caller, context_code, file_id = ds[idx]
                a = by_file[names[file_id]]
                self.assertEqual(caller, a.emitter)
                self.assertEqual(context_code, a.context)

    def test_windows_never_span_files(self):
        # windows are built per file, so file_id must be non-decreasing across
        # the dataset and each file's window count must match the per-file
        # arithmetic (a straddling window would break both).
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            t = np.arange(int(0.5 * SR)) / SR
            ann = []
            for i, c in enumerate([1, 2, 3], start=1):
                name = f"f{i}.wav"
                sf.write(
                    td / name, (0.1 * c + 0.01 * np.sin(2 * np.pi * 100 * t)).astype(np.float32), SR
                )
                ann.append(CallAnnotation(name, c, c, 1, 0, 0, 0, 0))
            ds = self._dataset(td, ann)
            ids = [ds[idx][4] for idx in range(len(ds))]
            self.assertEqual(ids, sorted(ids))
            counts: dict = {}
            for fid in ids:
                counts[fid] = counts.get(fid, 0) + 1
            self.assertEqual(sorted(counts.values()), [3, 3, 3])  # per-file arithmetic

    def test_split_filtering(self):
        with tempfile.TemporaryDirectory() as td:
            ann = _write_bouts(Path(td))
            keep = {"01_0.wav", "02_1.wav"}
            ds = self._dataset(Path(td), ann, split_files=keep)
            self.assertEqual(sorted(ds.file_names), sorted(keep))
            self.assertEqual(len(ds), 6)  # 2 files x 3 windows

    def test_drop_labels_arity(self):
        with tempfile.TemporaryDirectory() as td:
            ann = _write_bouts(Path(td))
            ds = self._dataset(Path(td), ann, drop_labels=True)
            item = ds[0]
            self.assertEqual(len(item), 2)

    def test_transform_applied_before_context_future_cut(self):
        # zeroing transform must zero BOTH context and future (noise consistent
        # across the prediction boundary)
        with tempfile.TemporaryDirectory() as td:
            ann = _write_bouts(Path(td))
            ds = self._dataset(Path(td), ann, transform=lambda frames: frames * 0.0)
            ctx, fut = ds[0][0], ds[0][1]
            self.assertEqual(float(ctx.abs().sum()), 0.0)
            self.assertEqual(float(fut.abs().sum()), 0.0)

    def test_short_file_yields_no_windows(self):
        # files shorter than sequence_length+steps_ahead are skipped (the
        # parent arithmetic gives 0 windows; padded windows would be mostly
        # zeros — training noise, deliberately excluded)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            t = np.arange(int(0.12 * SR)) / SR  # 12 frames < seq_len 20
            sf.write(td / "tiny.wav", np.sin(2 * np.pi * 300 * t).astype(np.float32), SR)
            ann = [CallAnnotation("tiny.wav", 1, 1, 1, 0, 0, 0, 0)]
            ds = self._dataset(td, ann)
            self.assertEqual(len(ds), 0)
            self.assertEqual(ds.file_names, [])


if __name__ == "__main__":
    unittest.main()
