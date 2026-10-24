#!/usr/bin/env python3
"""Two-arm vocal-target CPC pilot (§46 follow-on, bounded).

Arm A (reference): existing corrected objective on valid observed audio.
Arm B (vocal-target): same model/setup, vocal-eligible targets only.
Both continue from the SAME parent checkpoint (bat_cpc_v2, self-supervised —
NOT the caller-supervised multihead) with the SAME additional update budget.
Endpoints fixed in embeddings.vocal_target.two_arm_pilot_spec(); training
InfoNCE is explicitly NOT an endpoint.

Usage:
    python scripts/pilot_vocal_target.py --arm reference --device xpu:0
    python scripts/pilot_vocal_target.py --arm vocal --device xpu:0
    python scripts/pilot_vocal_target.py --self-test

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["reference", "vocal"], required=True)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="supervisor compatibility: accepted but the pilot is fresh-init "
        "by design (an interrupted arm restarts from its declared parent "
        "checkpoint — no mid-run state exists to resume)",
    )
    ap.add_argument("--parent", default="bat_cpc_v2")
    ap.add_argument(
        "--exclude-manifest",
        type=str,
        default=None,
        help="JSON with a 'files' list to EXCLUDE from the training window "
        "pool (fail-fast: evaluation files never enter any arm's training)",
    )
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        args.updates = 8
        args.batch_size = 4

    import numpy as np
    import torch

    from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
    from bioacoustic_pipeline.embeddings.cpc_trainer import CPCModel, TrainingConfig
    from bioacoustic_pipeline.embeddings.vocal_target import (
        two_arm_pilot_spec,
        vocal_mask_from_audio,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        parse_annotations_csv,
    )
    from bioacoustic_pipeline.utils.audio import load_audio
    from bioacoustic_pipeline.utils.devices import resolve_device

    device = resolve_device(args.device)
    print(f"pilot arm={args.arm} device={device}", flush=True)

    # ---- corpus (shared with §45/§46 splits; eval cohort excluded) ----

    if args.self_test:
        import soundfile as sf
        import tempfile

        td = tempfile.TemporaryDirectory()
        td_path = Path(td.name)
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
        import csv as _csv

        rows = []
        sr = 8000
        t = np.arange(int(0.8 * sr)) / sr
        for c in range(1, 5):
            for f in range(6):
                name = f"{c:02d}_{f}.wav"
                sig = np.concatenate(
                    [
                        np.zeros(int(0.2 * sr)),
                        0.5 * np.sin(2 * np.pi * (400 + 100 * c) * t[: int(0.4 * sr)]),
                        np.zeros(int(0.2 * sr)),
                    ]
                )
                sf.write(td_path / name, sig.astype(np.float32), sr)
                rows.append(dict(zip(cols, [name, c, 1, 1, 0, 0, 0, 0])))
        csv_path = td_path / "annotations.csv"
        with open(csv_path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        from bioacoustic_pipeline.ingestion.labeled_corpus import parse_annotations_csv

        annotations = parse_annotations_csv(csv_path)
        audio_root = td_path
        parent_ckpt = None
    else:
        annotations = [
            a
            for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
            if a.emitter > 0
        ]
        audio_root = ROOT / "data/egyptian_fruit_bats/audio"
        parent_ckpt = ROOT / "checkpoints" / args.parent / "best_model.pt"

    # ---- model: shared parent ----
    cfg = TrainingConfig(
        sample_rate=250000 if not args.self_test else 8000,
        frame_size_ms=10,
        hidden_dim=32 if args.self_test else 128,
        steps_ahead=4 if not args.self_test else 2,
        sequence_length=(8 if args.self_test else 64),
        overlap=(2 if args.self_test else 8),
        batch_size=args.batch_size,
        learning_rate=1e-5,  # continuation: gentle
        temperature=0.07,
        negative_samples=10,
        num_epochs=1,
        gradient_clip=1.0,
        device=device,
    )
    model = CPCModel(
        encoder_config=EncoderConfig(
            sample_rate=cfg.sample_rate, frame_size_ms=10, hidden_dim=cfg.hidden_dim
        ),
        steps_ahead=cfg.steps_ahead,
    )
    if parent_ckpt is not None and parent_ckpt.exists():
        ck = torch.load(parent_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        parent_hash = hashlib.sha256(parent_ckpt.read_bytes()).hexdigest()[:16]
        parent_loss_version = ck.get(
            "loss_version",
            "LEGACY (pre-v2 objective — positive could appear "
            "among negatives; both arms inherit this history identically)",
        )
        print(
            f"parent loaded: {args.parent} hash {parent_hash} objective={parent_loss_version}",
            flush=True,
        )
    else:
        parent_hash = "synthetic_selftest"
        parent_loss_version = "synthetic"
    model = model.to(device)

    # ---- window pool with vocal flags ----
    excluded = set()
    if args.exclude_manifest:
        import json as _json

        _m = _json.loads(Path(args.exclude_manifest).read_text())
        excluded = set(_m.get("files", []))
        print(
            f"EXCLUSION: {len(excluded)} evaluation files excluded from training pool", flush=True
        )
    frame = int(cfg.sample_rate * 0.01)
    windows: list = []  # (audio_window, vocal_flag)
    for a in annotations:
        if a.file_name in excluded:
            continue  # evaluation file: NEVER enters training pool
        p = audio_root / a.file_name
        if not p.exists():
            continue
        audio, sr = load_audio(str(p), target_sr=cfg.sample_rate)
        peak = float(np.abs(audio).max()) or 1.0
        audio = audio / peak
        n = len(audio) // frame
        vf = vocal_mask_from_audio(audio, sr, frame_ms=10, threshold_db=-36.0)
        seq = cfg.sequence_length + cfg.steps_ahead
        stride = max(cfg.sequence_length - cfg.overlap, 1)
        for s in range(0, max(n - seq, 0) + 1, stride * 3):  # stride*3: pilot budget
            w = audio[s * frame : (s + seq) * frame]
            if len(w) < seq * frame:
                continue
            mid = vf[s : s + seq]
            wf = w.reshape(seq, frame)  # (T, F) per-window framing for the model
            windows.append((wf, bool(mid.sum() >= len(mid) // 2)))
        if len(windows) > (4000 if not args.self_test else 64):
            break
    print(f"window pool: {len(windows)} | vocal {sum(v for _, v in windows)}", flush=True)

    # ---- matched training: same budget (updates), same seed ----
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model.train()
    losses = []
    from bioacoustic_pipeline.embeddings.vocal_target import safe_vocal_train_step

    completed, skipped = 0, 0
    for step in range(args.updates):
        idx = rng.randint(0, len(windows), args.batch_size)
        batch = np.stack([windows[i][0] for i in idx])
        x = torch.from_numpy(batch.astype(np.float32)).to(device)
        if args.arm == "reference":
            vocal = torch.ones(
                args.batch_size, cfg.sequence_length + cfg.steps_ahead, dtype=torch.bool
            )
        else:
            vf = torch.tensor([windows[i][1] for i in idx])
            vocal = (
                vf.unsqueeze(1)
                .expand(args.batch_size, cfg.sequence_length + cfg.steps_ahead)
                .contiguous()
            )
        outcome = safe_vocal_train_step(
            model, opt, x, vocal, temperature=cfg.temperature, n_negatives=cfg.negative_samples
        )
        if outcome["completed_update"]:
            losses.append(outcome["loss"])
            completed += 1
        else:
            skipped += 1  # attempted but NOT a successful update
        if (step + 1) % 100 == 0 and losses:
            print(
                f"  step {step + 1}/{args.updates} loss {np.mean(losses[-100:]):.4f}"
                f" | completed {completed} skipped {skipped}",
                flush=True,
            )

    # ---- save pilot checkpoint with provenance ----
    out_dir = ROOT / "checkpoints" / f"pilot_{args.arm}_vocaltarget"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arm": args.arm,
            "parent": args.parent,
            "parent_hash": parent_hash,
            "provenance": (
                "caller-supervised-derived continuation model"
                if args.parent == "bat_cpc_multihead_b32"
                else "self-supervised species checkpoint (legacy-objective parent)"
            ),
            "parent_loss_version": parent_loss_version,
            "budget": {"attempted": args.updates, "completed": completed, "skipped": skipped},
            "updates": args.updates,
            "seed": args.seed,
            "spec": two_arm_pilot_spec(),
            "loss_version": "v2-positive-exclusion-both-branches+valid-anchor-reduction",
        },
        out_dir / "pilot_checkpoint.pt",
    )
    payload = {
        "arm": args.arm,
        "parent": args.parent,
        "parent_hash": parent_hash,
        "provenance": (
            "caller-supervised-derived continuation model"
            if args.parent == "bat_cpc_multihead_b32"
            else "self-supervised species checkpoint (legacy-objective parent)"
        ),
        "parent_loss_version": parent_loss_version,
        "budget": {"attempted": args.updates, "completed": completed, "skipped": skipped},
        "updates": args.updates,
        "n_windows": len(windows),
        "final_loss_mean": round(float(np.mean(losses[-100:])), 5),
        "note": "training InfoNCE is NOT an endpoint (different target mixes "
        "change task difficulty); endpoints in two_arm_pilot_spec()",
    }
    (ROOT / "outputs" / f"pilot_{args.arm}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)
    if args.self_test:
        assert losses and np.isfinite(losses).all()
        print("SELF-TEST OK [{args.arm}]".replace("{args.arm}", args.arm), flush=True)


if __name__ == "__main__":
    main()
