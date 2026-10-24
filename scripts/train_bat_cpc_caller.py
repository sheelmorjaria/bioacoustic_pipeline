#!/usr/bin/env python3
"""Train bat_cpc_caller — Arm B (caller-conditioned) of §45.

SUPERVISED metric learning, not self-supervised representation: caller IDs
enter the contrastive loss (caller embedding pre-AR + same-caller-different-
file positives). Registered in the firewall as ``contains_caller_supervision:
true`` with ``allowed_use: [probe, retrieval]`` — it can never be evidence
for caller-structure claims (circular) and the EmbeddingBank hard-fails it.

Emitter ids are remapped to contiguous 0-based indices for the embedding
table; the mapping + n_callers are stored in the checkpoint so the backbone
can decode caller indices later.

Usage:
    python scripts/train_bat_cpc_caller.py --data-dir data/egyptian_fruit_bats \
        [--epochs 40] [--lambda-caller 0.5] [--caller-temperature 0.1]
    python scripts/train_bat_cpc_caller.py --self-test

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_bat_cpc as ARM_A  # noqa: E402  (shared corpus/config helpers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "egyptian_fruit_bats")
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--holdout-callers", type=int, default=10)
    ap.add_argument("--lambda-caller", type=float, default=0.5)
    ap.add_argument("--caller-temperature", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import dataclasses

    import torch

    from bioacoustic_pipeline.embeddings.caller_conditioned import (
        CallerCPCTrainer,
        CallerConditionedCPC,
    )
    from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
    from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
        LabeledAudioSequenceDataset,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        holdout_callers,
        parse_annotations_csv,
        split_files_by_caller,
    )
    from bioacoustic_pipeline.utils.devices import resolve_device

    selftest = args.self_test
    if selftest:
        args.epochs = 3
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
            audio_dir, annotations_csv = ARM_A._selftest_corpus(Path(td))
        else:
            audio_dir = args.data_dir / "audio"
            annotations_csv = args.annotations or (args.data_dir / "annotations.csv")
            if not audio_dir.exists() or not Path(annotations_csv).exists():
                sys.exit(f"data not found under {args.data_dir}; use --self-test offline")
        annotations = parse_annotations_csv(annotations_csv)
        n_all = len(annotations)
        annotations = [a for a in annotations if a.emitter > 0]
        print(f"identified-caller bouts: {len(annotations)}/{n_all}", flush=True)

        try:
            split = holdout_callers(
                annotations, n_heldout=args.holdout_callers, min_files=8, seed=args.seed
            )
            train_files, val_files = split.train_files, split.val_files
        except ValueError as exc:
            print(f"holdout split unavailable ({exc}); falling back to file split", flush=True)
            s = split_files_by_caller(annotations, seed=args.seed)
            train_files, val_files = s["train"], s["val"]

        train_emitters = sorted({a.emitter for a in annotations if a.file_name in train_files})
        emitter_to_idx = {e: i for i, e in enumerate(train_emitters)}
        remapped = [
            dataclasses.replace(a, emitter=emitter_to_idx[a.emitter])
            if a.file_name in train_files or a.file_name in val_files
            else a
            for a in annotations
        ]
        print(
            f"corpus: {len(annotations)} bouts | train {len(train_files)} / val {len(val_files)} "
            f"| {len(train_emitters)} training callers",
            flush=True,
        )

        config = ARM_A._bat_config(args, selftest=selftest)
        config.device = device
        resume_path = None
        if args.resume:
            cands = sorted(
                Path(config.checkpoint_dir).glob("checkpoint_epoch_*.pt"),
                key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
            )
            resume_path = str(cands[-1]) if cands else None
        config.checkpoint_dir = str(
            ROOT / "checkpoints" / ("bat_cpc_caller_selftest" if selftest else "bat_cpc_caller")
        )
        torch.manual_seed(args.seed)

        train_ds = LabeledAudioSequenceDataset(
            audio_dir, remapped, config, split_files=train_files, drop_labels=False
        )
        val_ds = LabeledAudioSequenceDataset(
            audio_dir, remapped, config, split_files=val_files, drop_labels=False
        )
        if len(train_ds) == 0:
            sys.exit("no trainable windows")
        print(f"windows: train={len(train_ds)} val={len(val_ds)}", flush=True)

        model = CallerConditionedCPC(
            encoder_config=EncoderConfig(
                sample_rate=config.sample_rate,
                frame_size_ms=config.frame_size_ms,
                hidden_dim=config.hidden_dim,
            ),
            steps_ahead=config.steps_ahead,
            n_callers=len(train_emitters),
        ).to(device)
        trainer = CallerCPCTrainer(
            model,
            config,
            train_dataset=train_ds,
            val_dataset=val_ds or None,
            lambda_caller=args.lambda_caller,
            caller_temperature=args.caller_temperature,
            n_callers=len(train_emitters),
        )
        if resume_path:
            trainer.load_checkpoint(resume_path)
            print(f"resumed from {resume_path} (epoch {trainer.current_epoch})", flush=True)
        history = trainer.train()
        # save provenance alongside the trainer checkpoint (trainer writes
        # best_model.pt; this adds the mapping the backbone needs)
        import json

        provenance = {
            "n_callers": len(train_emitters),
            "emitter_to_idx": {str(k): v for k, v in emitter_to_idx.items()},
            "lambda_caller": args.lambda_caller,
            "caller_temperature": args.caller_temperature,
            "note": "caller-supervised; probe/retrieval only (see model_metadata.json)",
        }
        out_dir = Path(config.checkpoint_dir)
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
        losses = history["train_loss"]
        print(f"train_loss: first={losses[0]:.4f} last={losses[-1]:.4f}", flush=True)
        if selftest:
            ck = torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=False)
            ck["n_callers"] = len(train_emitters)
            torch.save(ck, out_dir / "best_model.pt")
            print("SELF-TEST OK — checkpoint in checkpoints/bat_cpc_caller_selftest/", flush=True)


if __name__ == "__main__":
    main()
