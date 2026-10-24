"""Labeled multi-file window dataset for species-CPC training on corpora.

Extends the single-file ``AudioSequenceDataset`` windowing to a whole
annotated corpus (one ``CallAnnotation`` per WAV): the same frame/stride
arithmetic, but windows are built per file — a window NEVER spans two
recordings — and each window carries its file's caller/context labels.

``drop_labels=True`` yields exactly the parent's ``(context, future)`` tuple,
so the stock ``CPCTrainer`` (which unpacks two elements) trains Arm A
(self-supervised) with labels mechanically absent from the loss path. With
``drop_labels=False`` the tuple grows to
``(context, future, caller_id, context_code, file_id)`` for the
caller-conditioned arm. This arity switch IS the leakage guard for Arm A.

``transform`` (e.g. ``embeddings.augment.add_white_noise``) is applied to the
whole frame block BEFORE the context/future cut, so injected noise is
consistent across the prediction boundary.

Audio is loaded lazily per file (the full corpus does not fit in RAM) with a
one-file cache — windows are ordered by file, so consecutive access reuses it.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from bioacoustic_pipeline.embeddings.cpc_trainer import TrainingConfig
from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation
from bioacoustic_pipeline.utils.audio import load_audio

logger = logging.getLogger(__name__)


class LabeledAudioSequenceDataset(Dataset):
    """CPC windows over a labeled corpus: (context, future[, labels...])."""

    def __init__(
        self,
        audio_dir: str | Path,
        annotations: list[CallAnnotation],
        config: TrainingConfig,
        split_files: set[str] | None = None,
        *,
        drop_labels: bool = False,
        transform=None,
        preload: bool = False,
    ):
        self.config = config
        self.drop_labels = drop_labels
        self.transform = transform
        self.frame_size = int(config.sample_rate * config.frame_size_ms / 1000)
        seq_len = config.sequence_length + config.steps_ahead
        self._seq_len = seq_len
        self._stride = max(1, config.sequence_length - config.overlap)

        audio_dir = Path(audio_dir)
        wanted = set(split_files) if split_files is not None else None
        # Items are ANNOTATIONS (intervals), audio is cached per RECORDING:
        # interval corpora (marmoset) annotate both twins' calls inside one
        # shared wav, and the recording — not the interval — is the leakage
        # unit (one mic per file), so file ids are per-recording.
        self._recordings: list[str] = []  # recording name -> idx (file_id)
        self._recording_idx: dict[str, int] = {}
        self._paths: dict[str, Path] = {}
        self._items: list[tuple[CallAnnotation, int, int]] = []  # (ann, file_id, base_frame)
        for a in sorted(annotations, key=lambda x: (x.file_name, x.start_ms or 0.0, x.emitter)):
            key = a.recording_group or a.file_name
            if wanted is not None and key not in wanted:
                continue
            path = audio_dir / a.file_name
            if not path.exists():
                logger.warning("annotation without audio, skipped: %s", path)
                continue
            base_ms = a.start_ms or 0.0
            duration_ms = (a.end_ms - a.start_ms) if a.end_ms is not None else None
            if duration_ms is None:
                n_frames = self._count_frames(path)
            else:
                n_frames = int(duration_ms * self.config.sample_rate / 1000.0 / self.frame_size)
            n_windows = max(0, (n_frames - self._seq_len) // self._stride + 1)
            if n_windows == 0:
                continue  # too short to window — excluded by design
            if a.file_name not in self._recording_idx:
                self._recordings.append(a.file_name)
                self._recording_idx[a.file_name] = len(self._recordings) - 1
                self._paths[a.file_name] = path
            file_id = self._recording_idx[a.file_name]
            base_frame = int(base_ms * self.config.sample_rate / 1000.0 / self.frame_size)
            for w in range(n_windows):
                self._items.append((a, file_id, base_frame, w))
        self._cache: tuple[str, np.ndarray] | None = None
        self._preloaded: dict[str, np.ndarray] | None = None
        if preload:
            # Long-recording corpora defeat any per-item cache under shuffle
            # (each window would re-load a multi-MB wav). Framing everything
            # once into RAM (~GBs) makes __getitem__ pure slicing; forked
            # DataLoader workers share the pages copy-on-write.
            self._preloaded = {}
            for name, path in self._paths.items():
                audio = self._load(path)
                n = len(audio) // self.frame_size
                self._preloaded[name] = audio[: n * self.frame_size].reshape(n, self.frame_size)
            logger.info("preloaded %d framed recordings into RAM", len(self._preloaded))

    @property
    def file_names(self) -> list[str]:
        """Recording names in first-seen order (back-compat for file-level maps)."""
        return list(self._recordings)

    def _count_frames(self, path: Path) -> int:
        audio = self._load(path)
        return len(audio) // self.frame_size

    def _load(self, path: Path) -> np.ndarray:
        audio, sr = load_audio(str(path), target_sr=self.config.sample_rate)
        audio = np.asarray(audio, dtype=np.float64)
        peak = np.abs(audio).max() if audio.size else 0.0
        if peak > 0:
            audio = audio / peak
        return audio

    def __len__(self) -> int:
        return len(self._items)

    def _frames_for(self, file_name: str) -> np.ndarray:
        if self._preloaded is not None:
            return self._preloaded[file_name]
        if self._cache is None or self._cache[0] != file_name:
            audio = self._load(self._paths[file_name])
            num_frames = len(audio) // self.frame_size
            frames = audio[: num_frames * self.frame_size].reshape(num_frames, self.frame_size)
            self._cache = (file_name, frames)
        return self._cache[1]

    def __getitem__(self, idx: int):
        ann, file_id, base_frame, window = self._items[idx]
        frames = self._frames_for(ann.file_name)
        start = base_frame + window * self._stride
        block = frames[start : start + self._seq_len]
        if len(block) < self._seq_len:  # interval tail: zero-pad
            pad = np.zeros((self._seq_len - len(block), self.frame_size))
            block = np.concatenate([block, pad], axis=0)
        if self.transform is not None:
            block = self.transform(block)
        context = block[: self.config.sequence_length]
        future = block[self.config.sequence_length :]
        ctx = torch.from_numpy(np.ascontiguousarray(context)).float()
        fut = torch.from_numpy(np.ascontiguousarray(future)).float()
        if self.drop_labels:
            return ctx, fut
        return ctx, fut, ann.emitter, ann.context, file_id
