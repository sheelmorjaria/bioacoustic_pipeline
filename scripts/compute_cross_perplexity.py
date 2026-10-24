#!/usr/bin/env python3
"""Compute the ΔL_CPC cross-perplexity landscape for a recording (§46 Ph2).

Loads a trained SharedEncoderMultiHeadCPC checkpoint, frames a wav at the
model's rate, and writes the temporally-aligned donor-vs-own loss-difference
series (one value per valid context frame). Output: CSV with columns
frame_idx, start_ms, delta_L. Checkpoint-gated; --self-test uses a synthetic
untrained model on synthetic audio (mechanics only).

Usage:
    python scripts/compute_cross_perplexity.py --checkpoint marmoset_cpc_multihead \
        --audio data/alp/infant_marmosets_vox/audio_16k/twin_1/XXX.wav \
        --donor 1 --recipient 0
    python scripts/compute_cross_perplexity.py --self-test

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="marmoset_cpc_multihead")
    ap.add_argument("--audio", type=Path, default=None)
    ap.add_argument("--donor", type=int, required="--self-test" not in sys.argv)
    ap.add_argument("--recipient", type=int, default=None)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch

    from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
    from bioacoustic_pipeline.embeddings.multi_head_cpc import (
        SharedEncoderMultiHeadCPC,
        cross_perplexity_series,
    )
    from bioacoustic_pipeline.utils.devices import resolve_device

    device = resolve_device(args.device)
    if args.self_test:
        torch.manual_seed(0)
        model = SharedEncoderMultiHeadCPC(
            encoder_config=EncoderConfig(
                sample_rate=8000,
                frame_size_ms=10,
                hidden_dim=16,
                num_channels=(8, 16, 16),
                kernel_sizes=(3, 3, 3),
            ),
            steps_ahead=2,
            n_callers=3,
        ).to(device)
        g = torch.Generator().manual_seed(1)
        audio = torch.randn(1, 60, 80, generator=g).to(device)
        donor, recipient, sr = 1, 0, 8000
    else:
        ckpt_path = ROOT / "checkpoints" / args.checkpoint / "best_model.pt"
        if not ckpt_path.exists():
            sys.exit(f"checkpoint missing: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        steps = len({k.split(".")[1] for k in sd if k.startswith("heads.")} - {"unk"})
        hidden = sd["encoder.encoder.8.weight"].shape[0]
        model = SharedEncoderMultiHeadCPC(
            encoder_config=EncoderConfig(
                sample_rate=args.sample_rate, frame_size_ms=10, hidden_dim=hidden
            ),
            steps_ahead=max(steps, 2),
            n_callers=ckpt.get("n_callers", steps),
        )
        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()
        from bioacoustic_pipeline.utils.audio import load_audio

        audio_np, sr = load_audio(str(args.audio), target_sr=args.sample_rate)
        frame = int(sr * 0.01)
        n = len(audio_np) // frame
        peak = np.abs(audio_np[: n * frame]).max() or 1.0
        frames = (audio_np[: n * frame].reshape(n, frame) / peak).astype(np.float32)
        audio = torch.from_numpy(frames).unsqueeze(0).to(device)
        donor, recipient = args.donor, args.recipient

    series = (
        cross_perplexity_series(model, audio, donor_caller=donor, recipient_caller=recipient)
        .cpu()
        .numpy()
    )
    hop_ms = 10.0
    out = args.out or (ROOT / "outputs" / "cross_perplexity_selftest.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame_idx", "start_ms", "delta_L"])
        w.writeheader()
        for i, v in enumerate(series):
            w.writerow({"frame_idx": i, "start_ms": round(i * hop_ms, 1), "delta_L": float(v)})
    stats = {
        "n_frames": int(len(series)),
        "delta_mean": float(series.mean()),
        "delta_max": float(series.max()),
        "donor": donor,
        "recipient": recipient,
        "framing": "diagnostic caller-mismatch measurement, NOT meaning",
    }
    print(json.dumps(stats), flush=True)
    print(f"wrote {out}", flush=True)
    if args.self_test:
        assert len(series) == 60 - model.steps_ahead
        print("SELF-TEST OK", flush=True)


if __name__ == "__main__":
    main()
