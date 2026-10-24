#!/usr/bin/env python3
"""Train bat_cpc_v2 — Arm A (self-supervised) of §45.

Caller labels are MECHANICALLY excluded from training: the labeled dataset is
built with ``drop_labels=True`` and fed to the stock ``CPCTrainer``, whose
loss path never sees a label tensor. The InfoNCE negative cap
(``negative_samples``) is wired through (§45 M8) — without it training stalls
at the ln(B·T) random floor.

Training corpus: the public Egyptian fruit bat corpus (archive.org,
egyptian_fruit_bats_202102). ``--self-test`` runs the whole path on synthetic
bat-rate FM-sweep bouts with no data and 2 epochs.

Usage:
    python scripts/train_bat_cpc.py --data-dir data/egyptian_fruit_bats \
        [--epochs 40] [--batch-size 64] [--max-files N] [--holdout-callers 10]
    python scripts/train_bat_cpc.py --self-test

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _bat_config(args, *, selftest: bool = False):
    from bioacoustic_pipeline.embeddings.cpc_trainer import TrainingConfig

    return TrainingConfig(
        sample_rate=250000,
        frame_size_ms=10,
        hidden_dim=32 if selftest else 128,
        steps_ahead=4,
        sequence_length=32 if selftest else 64,
        overlap=8,
        batch_size=args.batch_size,
        learning_rate=2e-5,
        temperature=0.07,
        negative_samples=10,
        num_epochs=args.epochs,
        gradient_clip=1.0,
        checkpoint_dir=str(
            ROOT / "checkpoints" / ("bat_cpc_v2_selftest" if selftest else "bat_cpc_v2")
        ),
        device=args.device or "",
        save_every=1,  # epoch-level checkpoints: resumable on flaky GPUs
    )


def _selftest_corpus(td: Path, n_callers: int = 6, files_per_caller: int = 9):
    """Synthetic bat-rate FM-sweep bouts + annotations.csv (the real schema)."""
    import numpy as np
    import soundfile as sf

    rng = np.random.RandomState(0)
    audio = td / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    cols = [
        "File Name",
        "Emitter",
        "Addressee",
        "Context",
        "Emitter pre-vocalization action",
        "Addressee pre-vocalization action",
        "Emitter post-vocalization action",
        "Addressee post-vocalization action",
    ]
    rows = []
    sr = 250000
    for c in range(1, n_callers + 1):
        for f in range(files_per_caller):
            dur = rng.uniform(0.5, 0.9)  # 50-90 frames; seq_len=36 -> 1-3 windows
            t = np.arange(int(dur * sr)) / sr
            f0 = 20000 + 4000 * c + 2000 * f
            sweep = 0.5 * np.sin(2 * np.pi * (f0 * t + (f0 / (2 * dur)) * t**2))
            name = f"{c:02d}_{f}.wav"
            sf.write(audio / name, sweep.astype(np.float32), sr)
            rows.append({c2: v for c2, v in zip(cols, [name, c, 1, 1 + (f % 3), 0, 0, 0, 0])})
    csv_path = td / "annotations.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    return audio, csv_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "egyptian_fruit_bats")
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=None, help="budget: stratified subset")
    ap.add_argument("--holdout-callers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch

    from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
    from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel, CPCTrainer
    from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
        LabeledAudioSequenceDataset,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        holdout_callers,
        parse_annotations_csv,
    )
    from bioacoustic_pipeline.utils.devices import resolve_device

    selftest = args.self_test
    if selftest:
        args.epochs = 2
        args.holdout_callers = 2

    if args.batch_size is None:
        args.batch_size = (
            4 if selftest else (64 if torch.cuda.is_available() or torch.xpu.is_available() else 8)
        )

    device = resolve_device(args.device)
    print(f"device={device}", flush=True)

    td_ctx = tempfile.TemporaryDirectory()
    with td_ctx as td:
        if selftest:
            audio_dir, annotations_csv = _selftest_corpus(Path(td))
        else:
            audio_dir = args.data_dir / "audio"
            annotations_csv = args.annotations or (args.data_dir / "annotations.csv")
            if not audio_dir.exists() or not Path(annotations_csv).exists():
                sys.exit(
                    f"data not found under {args.data_dir} (need audio/ + annotations.csv); "
                    "restore the corpus or use --self-test"
                )
        annotations = parse_annotations_csv(annotations_csv)
        # caller-ID protocol runs on identified callers only: negative Emitter
        # codes are dataset sentinels (unknown/uncategorized), not callers
        n_all = len(annotations)
        annotations = [a for a in annotations if a.emitter > 0]
        print(f"identified-caller bouts: {len(annotations)}/{n_all}", flush=True)
        if args.max_files and len(annotations) > args.max_files:
            import random

            rng = random.Random(args.seed)
            per = {}
            for a in annotations:
                per.setdefault(a.emitter, []).append(a.file_name)
            # keep the long tail: round-robin stratified subset
            picked: set = set()
            pools = {c: sorted(fs) for c, fs in per.items()}
            for c in pools:
                rng.shuffle(pools[c])
            while len(picked) < args.max_files and any(pools.values()):
                for c in list(pools):
                    if pools[c] and len(picked) < args.max_files:
                        picked.add(pools[c].pop())
            annotations = [a for a in annotations if a.file_name in picked]

        n_heldout = args.holdout_callers
        try:
            split = holdout_callers(annotations, n_heldout=n_heldout, min_files=8, seed=args.seed)
            train_files, val_files = split.train_files, split.val_files
            heldout_note = f" (few-shot holdout: {len(split.heldout_callers)} callers excluded)"
        except ValueError as exc:
            print(f"holdout split unavailable ({exc}); falling back to file split", flush=True)
            from bioacoustic_pipeline.ingestion.labeled_corpus import split_files_by_caller

            s = split_files_by_caller(annotations, seed=args.seed)
            train_files, val_files = s["train"], s["val"]
            heldout_note = ""
        print(
            f"corpus: {len(annotations)} bouts | train {len(train_files)} / val {len(val_files)}"
            f"{heldout_note}",
            flush=True,
        )

        config = _bat_config(args, selftest=selftest)
        config.device = device
        torch.manual_seed(args.seed)
        resume_path = None
        if args.resume:
            cands = sorted(
                Path(config.checkpoint_dir).glob("checkpoint_epoch_*.pt"),
                key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
            )
            resume_path = str(cands[-1]) if cands else None

        train_ds = LabeledAudioSequenceDataset(
            audio_dir, annotations, config, split_files=train_files, drop_labels=True
        )
        val_ds = LabeledAudioSequenceDataset(
            audio_dir, annotations, config, split_files=val_files, drop_labels=True
        )
        if len(train_ds) == 0:
            sys.exit("no trainable windows — check corpus/annotations")
        print(f"windows: train={len(train_ds)} val={len(val_ds)}", flush=True)

        model = CPCModel(
            encoder_config=EncoderConfig(
                sample_rate=config.sample_rate,
                frame_size_ms=config.frame_size_ms,
                hidden_dim=config.hidden_dim,
            ),
            steps_ahead=config.steps_ahead,
        ).to(device)
        trainer = CPCTrainer(model, config, train_dataset=train_ds, val_dataset=val_ds or None)
        if resume_path:
            trainer.load_checkpoint(resume_path)
            print(f"resumed from {resume_path} (epoch {trainer.current_epoch})", flush=True)
        history = trainer.train()
        train_losses = history["train_loss"]
        print(
            f"train_loss: first={train_losses[0]:.4f} last={train_losses[-1]:.4f} "
            f"best_val={min(history['val_loss']) if history['val_loss'] else float('nan'):.4f}",
            flush=True,
        )
        if selftest:
            assert train_losses[-1] < train_losses[0] + 1e-6, "self-test loss did not decrease"
            print("SELF-TEST OK — checkpoint in checkpoints/bat_cpc_v2_selftest/", flush=True)


if __name__ == "__main__":
    main()
