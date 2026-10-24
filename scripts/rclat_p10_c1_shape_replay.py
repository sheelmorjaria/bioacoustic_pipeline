"""Bounded synthetic replay of recorded training depthwise-convolution shapes.

No audio inference, checkpoint loading, CTC or optimizer updates. Validation
shapes and the full model's allocations are deliberately not reproduced.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch


def telemetry(device):
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.split(":")[0] in {"VmRSS", "VmSize", "VmSwap", "Threads"}:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    status.update(
        open_fds=len(list(Path("/proc/self/fd").iterdir())),
        memory_maps=len(Path("/proc/self/maps").read_text().splitlines()),
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        xpu_allocated=torch.xpu.memory_allocated(device),
        xpu_reserved=torch.xpu.memory_reserved(device),
        xpu_peak_allocated=torch.xpu.max_memory_allocated(device),
        xpu_peak_reserved=torch.xpu.max_memory_reserved(device),
    )
    try:
        free, total = torch.xpu.mem_get_info(device)
        status.update(xpu_free=free, xpu_total=total)
    except Exception as exc:
        status["device_free_memory_unavailable"] = str(exc)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--layers", type=int, default=12)
    args = parser.parse_args()
    out = Path("outputs/rclat_p10_c1_shape_replay")
    out.mkdir(exist_ok=True)
    target = out / f"xpu{args.device}.json"
    if target.exists():
        raise RuntimeError("Diagnostic result already exists; use a separate version")
    journal = Path("outputs/rclat_p10_corrected_v1/mel_steps.jsonl")
    other = Path("outputs/rclat_p10_corrected_v1/combined_steps.jsonl")
    sequence = [json.loads(s) for s in journal.read_text().splitlines() if '"event": "begin"' in s]
    second = [json.loads(s) for s in other.read_text().splitlines() if '"event": "begin"' in s]
    assert [r["uids"] for r in sequence] == [r["uids"] for r in second]
    lengths = {}
    for uid in {u for row in sequence for u in row["uids"]}:
        with np.load(f"cache/rclat_p10/{uid}.npz") as cache:
            lengths[uid] = len(cache["mel"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = f"xpu:{args.device}"
    torch.xpu.set_device(device)
    weights = [
        torch.randn(256, 1, 15, device=device, requires_grad=True) / 15**0.5
        for _ in range(args.layers)
    ]
    weights = [w.detach().requires_grad_() for w in weights]
    record = dict(
        torch=torch.__version__,
        device=device,
        layers=args.layers,
        journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
        identical_training_batch_order_verified=True,
        planned_steps=len(sequence),
        max_seconds=args.max_seconds,
        deterministic=True,
        scientific_training=False,
        optimizer_updates=0,
        validation_shapes_replayed=False,
        full_model_allocations_replayed=False,
        primitive_cache_occupancy="Not exposed by this probe; unique shapes are a proxy only",
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("ONEDNN_", "DNNL_", "SYCL_", "ZE_"))
        },
        snapshots=[],
        completed_steps=0,
    )
    start = time.monotonic()
    unique = set()
    with (out / f"xpu{args.device}_progress.jsonl").open("x", buffering=1) as progress:
        try:
            for row in sequence:
                if time.monotonic() - start >= args.max_seconds:
                    record["stop_reason"] = "wall_time_bound"
                    break
                frames = (max(lengths[u] for u in row["uids"]) + 3) // 4
                record["active_step"] = row["step"]
                record["active_shape"] = [len(row["uids"]), 256, frames]
                x = torch.randn(*record["active_shape"], device=device, requires_grad=True)
                y = x
                for weight in weights:
                    y = torch.nn.functional.conv1d(y, weight, groups=256, padding=7)
                y.square().mean().backward()
                if not torch.isfinite(y).all() or not torch.isfinite(x.grad).all():
                    raise FloatingPointError("Nonfinite diagnostic output/input gradient")
                if not all(bool(torch.isfinite(w.grad).all()) for w in weights):
                    raise FloatingPointError("Nonfinite diagnostic weight gradient")
                for weight in weights:
                    weight.grad = None
                del x, y
                record["completed_steps"] = row["step"]
                unique.add(frames)
                if row["step"] == 1 or row["step"] % 100 == 0 or row is sequence[-1]:
                    torch.xpu.synchronize(device)
                    snapshot = dict(
                        step=row["step"],
                        seconds=time.monotonic() - start,
                        unique_shapes=len(unique),
                        **telemetry(device),
                    )
                    record["snapshots"].append(snapshot)
                    progress.write(json.dumps(snapshot) + "\n")
                    print(json.dumps(snapshot), flush=True)
            else:
                record["stop_reason"] = "completed_historical_training_sequence"
        except Exception as exc:
            record.update(
                stop_reason="exception", error=repr(exc), failure_telemetry=telemetry(device)
            )
        record["elapsed_seconds"] = time.monotonic() - start
        record["historical_primitive_failure_reproduced"] = (
            "could not create a primitive" in record.get("error", "")
        )
        target.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "snapshots"}), flush=True)


if __name__ == "__main__":
    main()
