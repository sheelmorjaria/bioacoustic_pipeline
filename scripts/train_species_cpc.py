#!/usr/bin/env python3
"""Species-parameterized CPC training (§46): Arm A (self-supervised).

Generalizes scripts/train_bat_cpc.py beyond bat: ``--species marmoset``
switches the corpus parser (interval-windowed InfantMarmosetsVox at 16 kHz),
training config (seq 16+2, marmoset window arithmetic), and checkpoint name
(``marmoset_cpc_v2``). Caller labels are MECHANICALLY excluded from training
(drop_labels=True through the labeled dataset). Bat mode reproduces
train_bat_cpc.py exactly.

Usage:
    python scripts/train_species_cpc.py --species marmoset --self-test
    python scripts/train_species_cpc.py --species marmoset --device xpu:1
    python scripts/train_species_cpc.py --species bat --device xpu:1

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

import train_bat_cpc as BAT  # noqa: E402  (shared self-test corpus + supervisor resume)


SPECIES = {
    "bat": {
        "sample_rate": 250000,
        "sequence_length": 64,
        "steps_ahead": 4,
        "overlap": 8,
        "hidden_dim": 128,
        "default_data": ROOT / "data" / "egyptian_fruit_bats",
        "checkpoint": "bat_cpc_v2",
        "min_files": 8,
        "holdout": 10,
    },
    "marmoset": {
        "sample_rate": 16000,
        "sequence_length": 16,
        "steps_ahead": 2,
        "overlap": 2,
        "hidden_dim": 128,
        "default_data": ROOT / "data" / "alp" / "infant_marmosets_vox",
        # the mirror keeps bucket object paths: audio lives under data/…
        "audio_root": ROOT / "data" / "alp" / "infant_marmosets_vox" / "data",
        "checkpoint": "marmoset_cpc_v2",
        "min_files": 4,
        "holdout": 2,
    },
    "littleowl": {
        "sample_rate": 44100,
        "sequence_length": 64,
        "steps_ahead": 4,
        "overlap": 8,
        "hidden_dim": 128,
        "default_data": ROOT / "data" / "alp" / "littleowl_id" / "v0.1.0" / "raw",
        "checkpoint": "littleowl_cpc",
        "min_files": 3,
        "holdout": 1,
        "split_csv": "acrossyear_fg_train.csv",
    },
    "pipit": {
        "sample_rate": 44100,
        "sequence_length": 64,
        "steps_ahead": 4,
        "overlap": 8,
        "hidden_dim": 128,
        "default_data": ROOT / "data" / "alp" / "pipit_id" / "v0.1.0" / "raw",
        "checkpoint": "pipit_cpc",
        "min_files": 3,
        "holdout": 1,
        "split_csv": "acrossyear_fg_train.csv",
    },
    "chiffchaff": {
        "sample_rate": 44100,
        "sequence_length": 64,
        "steps_ahead": 4,
        "overlap": 8,
        "hidden_dim": 128,
        "default_data": ROOT / "data" / "alp" / "chiffchaff_id" / "v0.1.0" / "raw",
        "checkpoint": "chiffchaff_cpc",
        "min_files": 5,
        "holdout": 2,
        "split_csv": "acrossyear_fg_train.csv",
    },
}


def _config(args, sp, *, selftest: bool):
    from bioacoustic_pipeline.embeddings.cpc_trainer import TrainingConfig

    return TrainingConfig(
        sample_rate=sp["sample_rate"],
        frame_size_ms=10,
        hidden_dim=32 if selftest else sp["hidden_dim"],
        steps_ahead=sp["steps_ahead"],
        sequence_length=(8 if selftest else sp["sequence_length"]),
        overlap=sp["overlap"],
        batch_size=args.batch_size,
        learning_rate=2e-5,
        temperature=0.07,
        negative_samples=10,
        num_epochs=args.epochs,
        gradient_clip=1.0,
        checkpoint_dir=str(
            ROOT / "checkpoints" / (sp["checkpoint"] + ("_selftest" if selftest else ""))
        ),
        device=args.device or "",
        save_every=1,  # epoch-level checkpoints: resumable on flaky GPUs
        num_workers=0 if selftest else (8 if args.species == "marmoset" else 6),
    )


def _load_annotations(species: str, args, selftest: bool):
    if selftest:
        td = Path(tempfile.mkdtemp())
        if species == "bat":
            audio_dir, csv_path = BAT._selftest_corpus(td)
            from bioacoustic_pipeline.ingestion.labeled_corpus import parse_annotations_csv

            ann = parse_annotations_csv(csv_path)
            ann = [a for a in ann if a.emitter > 0]
            return audio_dir, ann
        # marmoset-style synthetic intervals
        import csv

        import numpy as np
        import soundfile as sf

        audio = td / "audio_16k"
        for tw in ("twin_1", "twin_2"):
            (audio / tw).mkdir(parents=True)
        cols = ["path", "start", "end", "duration", "calltypeID", "callerID", "twinID", "vocID"]
        rows = []
        for c in range(1, 5):
            for f in range(8):
                name = f"audio_16k/twin_{1 + c % 2}/t{c}_{f}.wav"
                t = np.arange(int(0.6 * 16000)) / 16000
                sig = 0.5 * np.sin(2 * np.pi * (300 + 250 * c) * t).astype(np.float32)
                sf.write(td / name, sig, 16000)
                rows.append(
                    [name, "0.0", "0.55", "0.55", str(1 + f % 3), str(c), str(1 + c % 2), str(f)]
                )
        csv_path = td / "labels.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(rows)
        from bioacoustic_pipeline.ingestion.labeled_corpus import parse_marmoset_labels_csv

        return td, parse_marmoset_labels_csv(csv_path)
    if species == "bat":
        from bioacoustic_pipeline.ingestion.labeled_corpus import parse_annotations_csv

        ann = parse_annotations_csv(args.annotations or (args.data_dir / "annotations.csv"))
        # audio lives under <root>/audio (the earlier return of the bare root
        # skipped every annotation -> "no trainable windows" crash loop)
        return args.data_dir / "audio", [a for a in ann if a.emitter > 0]
    if species == "littleowl":
        import csv as _csv

        from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation

        data_dir = args.data_dir
        id_map: dict = {}

        def _caller_int(id_str):
            if id_str not in id_map:
                id_map[id_str] = len(id_map) + 1
            return id_map[id_str]

        csv_name = SPECIES.get("littleowl", {}).get("split_csv", "acrossyear_fg_train.csv")
        csv_path = data_dir / csv_name
        anns = []
        if csv_path.exists():
            with open(csv_path) as fh:
                for row in _csv.DictReader(fh):
                    anns.append(
                        CallAnnotation(
                            file_name=row["local_path"],
                            emitter=_caller_int(row["individual_id"]),
                            addressee=0,
                            context=0,
                            emit_pre=0,
                            addr_pre=0,
                            emit_post=0,
                            addr_post=0,
                        )
                    )
        print(f"littleowl: {len(anns)} clips, {len(id_map)} callers", flush=True)
        return data_dir, anns

    if species == "pipit":
        import csv as _csv

        from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation

        data_dir = args.data_dir
        id_map: dict = {}

        def _caller_int(id_str):
            if id_str not in id_map:
                id_map[id_str] = len(id_map) + 1
            return id_map[id_str]

        csv_name = SPECIES.get("pipit", {}).get("split_csv", "withinyear_fg_train.csv")
        csv_path = data_dir / csv_name
        anns = []
        if csv_path.exists():
            with open(csv_path) as fh:
                for row in _csv.DictReader(fh):
                    anns.append(
                        CallAnnotation(
                            file_name=row["local_path"],
                            emitter=_caller_int(row["individual_id"]),
                            addressee=0,
                            context=0,
                            emit_pre=0,
                            addr_pre=0,
                            emit_post=0,
                            addr_post=0,
                        )
                    )
        print(f"pipit: {len(anns)} clips, {len(id_map)} callers", flush=True)
        return data_dir, anns

    if species == "chiffchaff":
        import csv as _csv

        from bioacoustic_pipeline.ingestion.labeled_corpus import CallAnnotation

        data_dir = args.data_dir
        id_map: dict = {}

        def _caller_int(id_str):
            if id_str not in id_map:
                id_map[id_str] = len(id_map) + 1
            return id_map[id_str]

        csv_name = SPECIES.get("chiffchaff", {}).get("split_csv", "withinyear_fg_train.csv")
        csv_path = data_dir / csv_name
        anns = []
        if csv_path.exists():
            with open(csv_path) as fh:
                for row in _csv.DictReader(fh):
                    anns.append(
                        CallAnnotation(
                            file_name=row["local_path"],
                            emitter=_caller_int(row["individual_id"]),
                            addressee=0,
                            context=0,
                            emit_pre=0,
                            addr_pre=0,
                            emit_post=0,
                            addr_post=0,
                        )
                    )
        print(
            f"chiffchaff: {len(anns)} clips, {len(id_map)} callers (ring-confirmed)",
            flush=True,
        )
        return data_dir, anns
    from bioacoustic_pipeline.ingestion.labeled_corpus import parse_marmoset_labels_csv

    ann = parse_marmoset_labels_csv(args.data_dir / "labels.csv")
    # labels reference the 44.1k masters; the corpus ships identical 16k
    # copies — train on those (native rate: no per-load resampling)
    import dataclasses

    ann = [
        dataclasses.replace(
            a,
            file_name=a.file_name.replace("audio_44k", "audio_16k"),
            recording_group=a.recording_group.replace("audio_44k", "audio_16k"),
        )
        for a in ann
    ]
    return args.data_dir, ann


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--species", choices=sorted(SPECIES), default="marmoset")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--holdout-callers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    sp = SPECIES[args.species]
    args.data_dir = args.data_dir or sp["default_data"]
    if args.holdout_callers is None:
        args.holdout_callers = sp["holdout"]
    selftest = args.self_test
    if selftest:
        args.epochs = 2
        args.holdout_callers = 1

    import torch

    from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
    from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel, CPCTrainer
    from bioacoustic_pipeline.embeddings.labeled_sequence_dataset import (
        LabeledAudioSequenceDataset,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        holdout_callers,
        split_files_by_caller,
    )
    from bioacoustic_pipeline.utils.devices import resolve_device

    if args.batch_size is None:
        args.batch_size = 4 if selftest else 64
    device = resolve_device(args.device)
    print(f"species={args.species} device={device}", flush=True)

    audio_root, annotations = _load_annotations(args.species, args, selftest)
    if not selftest and "audio_root" in sp:
        audio_root = sp["audio_root"]
    try:
        split = holdout_callers(
            annotations, n_heldout=args.holdout_callers, min_files=sp["min_files"], seed=args.seed
        )
        train_files, val_files = split.train_files, split.val_files
        heldout_note = f" (holdout: {len(split.heldout_callers)} callers)"
    except ValueError as exc:
        print(f"holdout unavailable ({exc}); file split fallback", flush=True)
        s = split_files_by_caller(annotations, seed=args.seed)
        train_files, val_files = s["train"], s["val"]
        heldout_note = ""
    print(
        f"corpus: {len(annotations)} items | train {len(train_files)} / val {len(val_files)}"
        f"{heldout_note}",
        flush=True,
    )

    config = _config(args, sp, selftest=selftest)
    config.device = device
    torch.manual_seed(args.seed)
    resume_path = None
    if args.resume:
        cands = sorted(
            Path(config.checkpoint_dir).glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        resume_path = str(cands[-1]) if cands else None

    # splits key on the leakage unit (recording group / file)
    preload = args.species == "marmoset" and not selftest
    train_ds = LabeledAudioSequenceDataset(
        audio_root,
        annotations,
        config,
        split_files=train_files,
        drop_labels=True,
        preload=preload,
    )
    val_ds = LabeledAudioSequenceDataset(
        audio_root,
        annotations,
        config,
        split_files=val_files,
        drop_labels=True,
        preload=preload,
    )
    if len(train_ds) == 0:
        sys.exit("no trainable windows")
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
    losses = history["train_loss"]
    print(
        f"train_loss: first={losses[0]:.4f} last={losses[-1]:.4f} "
        f"best_val={min(history['val_loss']) if history['val_loss'] else float('nan'):.4f}",
        flush=True,
    )
    if selftest:
        assert losses[-1] < losses[0] + 1e-6, "self-test loss did not decrease"
        print(f"SELF-TEST OK [{args.species}]", flush=True)


if __name__ == "__main__":
    main()
