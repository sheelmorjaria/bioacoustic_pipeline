#!/usr/bin/env python3
"""Species-parameterized multi-head CPC training (§46): the consolidated arm.

SharedEncoderMultiHeadCPC — shared grammar + per-caller prediction heads.
For marmoset this REPLACES the planned marmoset_cpc_caller run: one training
pass yields both the caller-conditioned arm and the Track-C cross-perplexity
instrument. Registered as ``marmoset_cpc_multihead`` with
``contains_caller_supervision: true`` (probe/retrieval only).

Usage:
    python scripts/train_species_cpc_multihead.py --species marmoset --self-test
    python scripts/train_species_cpc_multihead.py --species marmoset --device xpu:1

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_species_cpc as ARM_A  # noqa: E402  (shared species configs + loaders)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--species", choices=sorted(ARM_A.SPECIES), default="marmoset")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--holdout-callers", type=int, default=None)
    ap.add_argument("--lambda-heads", type=float, default=0.5)
    ap.add_argument(
        "--steps-ahead",
        type=int,
        default=None,
        help="override the species default (bat preregistration: 10-20)",
    )
    ap.add_argument(
        "--exclude-eval-cohort",
        action="store_true",
        help="drop the Stage 1-4 eval-sample bouts (cache/stage2) from training",
    )
    ap.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="",
        help="versioned output dir (e.g. _b40); parent state is never deleted",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch

    from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
    from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
        LabeledAudioSequenceDataset,
    )
    from bioacoustic_pipeline.embeddings.multi_head_cpc import (
        MultiHeadCPCTrainer,
        SharedEncoderMultiHeadCPC,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        holdout_callers,
        split_files_by_caller,
    )
    from bioacoustic_pipeline.utils.devices import resolve_device

    sp = dict(ARM_A.SPECIES[args.species])
    sp["checkpoint"] = sp["checkpoint"].replace("_v2", "_multihead")
    args.data_dir = args.data_dir or sp["default_data"]
    if args.holdout_callers is None:
        args.holdout_callers = sp["holdout"]
    selftest = args.self_test
    if selftest:
        args.epochs = 2
        args.holdout_callers = 1
    if args.batch_size is None:
        args.batch_size = 4 if selftest else 64
    device = resolve_device(args.device)
    print(f"species={args.species} multihead device={device}", flush=True)

    audio_root, annotations = ARM_A._load_annotations(args.species, args, selftest)
    if not selftest and "audio_root" in sp:
        audio_root = sp["audio_root"]  # marmoset: bucket object paths under data/
    if args.exclude_eval_cohort and not selftest:
        import numpy as np

        traj_path = ROOT / "cache" / "stage2" / "bat_c11c12_traj.npz"
        if traj_path.exists():
            traj = np.load(traj_path, allow_pickle=False)
            excluded = {str(n) for n, p in zip(traj["names"], traj["parts"]) if p == 1}
            before = len(annotations)
            annotations = [a for a in annotations if a.file_name not in excluded]
            print(
                f"eval-cohort exclusion: {before - len(annotations)} bouts dropped "
                f"({len(excluded)} sampled eval bouts)",
                flush=True,
            )
    try:
        split = holdout_callers(
            annotations,
            n_heldout=args.holdout_callers,
            min_files=sp["min_files"],
            seed=args.seed,
        )
        train_files, val_files = split.train_files, split.val_files
    except ValueError:
        s = split_files_by_caller(annotations, seed=args.seed)
        train_files, val_files = s["train"], s["val"]

    training = [a for a in annotations if (a.recording_group or a.file_name) in train_files]
    train_emitters = sorted({a.emitter for a in training})
    e2i = {e: i for i, e in enumerate(train_emitters)}
    import dataclasses

    remapped = [
        dataclasses.replace(a, emitter=e2i[a.emitter])
        if (a.recording_group or a.file_name) in train_files | val_files
        else a
        for a in annotations
    ]
    print(
        f"corpus: {len(annotations)} items | train {len(train_files)} / val {len(val_files)} "
        f"| {len(train_emitters)} training callers",
        flush=True,
    )

    config = ARM_A._config(args, sp, selftest=selftest)
    if args.steps_ahead:
        config.steps_ahead = args.steps_ahead
    config.device = device
    config.checkpoint_dir = str(
        ROOT
        / "checkpoints"
        / (sp["checkpoint"] + ("_selftest" if selftest else "") + args.checkpoint_suffix)
    )
    torch.manual_seed(args.seed)
    resume_path = None
    if args.resume:
        cands = sorted(
            Path(config.checkpoint_dir).glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        resume_path = str(cands[-1]) if cands else None

    train_ds = LabeledAudioSequenceDataset(
        audio_root, remapped, config, split_files=train_files, drop_labels=False
    )
    val_ds = LabeledAudioSequenceDataset(
        audio_root, remapped, config, split_files=val_files, drop_labels=False
    )
    if len(train_ds) == 0:
        sys.exit("no trainable windows")
    print(f"windows: train={len(train_ds)} val={len(val_ds)}", flush=True)

    model = SharedEncoderMultiHeadCPC(
        encoder_config=EncoderConfig(
            sample_rate=config.sample_rate,
            frame_size_ms=config.frame_size_ms,
            hidden_dim=config.hidden_dim,
        ),
        steps_ahead=config.steps_ahead,
        n_callers=len(train_emitters),
    ).to(device)
    print(
        f"steps_ahead={config.steps_ahead} seq={config.sequence_length} "
        f"n_callers={len(train_emitters)}",
        flush=True,
    )
    trainer = MultiHeadCPCTrainer(
        model,
        config,
        train_dataset=train_ds,
        val_dataset=val_ds or None,
        lambda_heads=args.lambda_heads,
    )
    if resume_path:
        trainer.load_checkpoint(resume_path)
        print(f"resumed from {resume_path} (epoch {trainer.current_epoch})", flush=True)
    history = trainer.train()
    losses = history["train_loss"]
    print(f"train_loss: first={losses[0]:.4f} last={losses[-1]:.4f}", flush=True)
    if selftest:
        assert losses[-1] < losses[0] + 1e-6
        print(f"SELF-TEST OK [{args.species} multihead]", flush=True)


if __name__ == "__main__":
    main()
