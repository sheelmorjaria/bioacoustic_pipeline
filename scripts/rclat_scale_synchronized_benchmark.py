"""Device-explicit timing correction for the new training-only engineering model."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import rclat_scale_engineering_v3 as engineering
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text, fit_scaler
from bioacoustic_pipeline.representation.rclat_scale_v3 import ScaleCTC


def run(arm, large=False):
    tag = "large_" if large else ""
    output = Path(f"outputs/rclat_scale_synchronized_{tag}{arm}.json")
    if output.exists():
        raise RuntimeError("Synchronized benchmark already exists")
    plan = engineering.check()
    kind, dimension, device = engineering.KINDS[arm]
    torch.xpu.set_device(device)
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    rows, cache = engineering.sources()
    mapping = {r["uid"]: r for r in rows}
    arrays = {r["uid"]: engineering.closed.sequence(cache[r["uid"]], kind) for r in rows}
    scaler = fit_scaler(arrays.values())
    uids = plan["throughput_uids"]
    arrays = {uid: frozen_scale(*arrays[uid], scaler) for uid in uids}
    lengths = torch.tensor([len(arrays[uid][0]) for uid in uids])
    cpu_values = pad_sequence([torch.from_numpy(arrays[uid][0]) for uid in uids], batch_first=True)
    cpu_masks = pad_sequence([torch.from_numpy(arrays[uid][1]) for uid in uids], batch_first=True)
    targets = [encode_text(mapping[uid]["transcript"]) for uid in uids]
    config = (
        {"width": 256, "layers": 12, "heads": 4, "expansion": 1024, "kernel": 15, "dropout": 0.0}
        if large
        else {"width": 128, "layers": 4, "heads": 4, "expansion": 512, "kernel": 15, "dropout": 0.0}
    )
    model = ScaleCTC(dimension, **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    def step():
        torch.xpu.synchronize(device)
        start = time.perf_counter()
        values = cpu_values.to(device)
        masks = cpu_masks.to(device)
        torch.xpu.synchronize(device)
        a = time.perf_counter()
        optimizer.zero_grad()
        logits, out_lengths = model(values, masks, lengths)
        torch.xpu.synchronize(device)
        b = time.perf_counter()
        loss = ctc_objective(logits, out_lengths, targets)
        c = time.perf_counter()
        loss.backward()
        torch.xpu.synchronize(device)
        d = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        torch.xpu.synchronize(device)
        end = time.perf_counter()
        return {
            "total_seconds": end - start,
            "input_transfer_seconds": a - start,
            "forward_seconds": b - a,
            "cpu_ctc_forward_seconds": c - b,
            "backward_including_cpu_ctc_seconds": d - c,
            "optimizer_and_clip_seconds": end - d,
        }

    for _ in range(2):
        step()
    torch.xpu.reset_peak_memory_stats(device)
    measurements = [step() for _ in range(10)]
    means = {key: float(np.mean([r[key] for r in measurements])) for key in measurements[0]}
    seconds = sum(mapping[uid]["duration_seconds"] for uid in uids)
    engineering.closed.prior.freeze_json(
        output,
        {
            "engineering_plan_sha256": engineering.closed.prior.sha(engineering.PLAN),
            "arm": arm,
            "device": device,
            "current_device": torch.xpu.current_device(),
            "explicit_device_synchronization": True,
            "warmup_steps": 2,
            "timed_steps": 10,
            "benchmark_mean": means,
            "audio_seconds_per_batch": seconds,
            "audio_seconds_per_wall_second": seconds / means["total_seconds"],
            "peak_allocated_bytes": torch.xpu.max_memory_allocated(device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "architecture_config": config,
            "precision": "FP32 neural and CPU CTC",
            "strict_deterministic_algorithms": True,
            "source_scope": "Same4long training utterances,precomputed features;excludes waveform extraction,disk I/O and larger-batch tuning.",
            "supersedes": "Earlier engineering timing used synchronize() without device argument;XPU1 timing is not used as a reliable phase breakdown.",
            "scientific_validation_or_test_scoring_performed": False,
            "stage2_ready": False,
        },
    )
    print(
        arm,
        "explicitly synchronized audioseconds/wallsecond",
        seconds / means["total_seconds"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--arm", choices=list(engineering.KINDS), required=True)
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run(args.arm, args.large)
