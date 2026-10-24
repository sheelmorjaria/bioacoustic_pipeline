"""Bounded 128-shape working-set test with a matched 12-shape control."""

import argparse
from contextlib import nullcontext
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from bioacoustic_pipeline.representation.rclat_scale_v4 import ScaleCTC
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective
from bioacoustic_pipeline.representation.rclat_numerics import (
    check_gradients,
    check_parameters_and_optimizer,
)
from rclat_p10_c1_shape_replay import telemetry
from rclat_p10_memory_attribution import Mallinfo2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["control12", "large128"], required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    out = Path("outputs/rclat_p10_large_shape_set")
    out.mkdir(exist_ok=True)
    name = f"{args.mode}_xpu{args.device}" + ("_profiled" if args.profile else "")
    planpath = out / f"{name}_plan.json"
    if planpath.exists():
        raise RuntimeError("Existing attempt; no retry")
    journal = Path("outputs/rclat_p10_corrected_v2/mel_steps.jsonl")
    unique = []
    seen = set()
    for line in journal.open():
        row = json.loads(line)
        if row["event"] not in ("forward_begin", "validation_forward_begin"):
            continue
        lens = row["native_lengths"]
        if len(lens) != 4 or min(lens) < 160:
            continue
        shape = (max(lens) + 3) // 4
        if shape not in seen:
            unique.append(lens)
            seen.add(shape)
        if len(unique) == 128:
            break
    assert len(unique) == 128
    selected = unique if args.mode == "large128" else [unique[i % 12] for i in range(128)]
    plan = dict(
        mode=args.mode,
        device=args.device,
        profile=args.profile,
        cycles=3,
        batches_per_cycle=128,
        updates=384,
        max_seconds=600,
        external_process_bound_seconds=660,
        lengths=selected,
        source_journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model="v4ScaleCTC64input,width256,12layers,4heads,expansion1024,kernel15;FP32deterministic",
        inputs="Synthetic CPU-generated features transferred to XPU;all-valid masks;40symbol synthetic targets;historical lengths only",
        optimizer="Fresh AdamW lr1e-5,wd0.01,clip5;finite guards;engineering updates only",
        selection="First128distinct subsampled max lengths from C2 training/validation metadata;batch4;each native length>=160 for synthetic CTC feasibility",
        profile_policy="Third cycle only;cycles1and2unprofiled;Memray1.20 native aggregated allocations"
        if args.profile
        else "none",
        limits="No real audio/transcript decoding,scientific scoring,checkpointIO,cache-policy change or complete historical workload;control matches update count,not FLOPs or length distribution",
    )
    planpath.write_text(json.dumps(plan, indent=2) + "\n")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = f"xpu:{args.device}"
    torch.xpu.set_device(device)
    libc = ctypes.CDLL(None)
    libc.mallinfo2.restype = Mallinfo2
    model = ScaleCTC(64, width=256, layers=12, heads=4, expansion=1024, kernel=15, dropout=0.0).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    targets = [torch.tensor([2, 3, 4, 5] * 10) for _ in range(4)]
    start = time.monotonic()
    record = dict(completed_updates=0, snapshots=[])
    progress = (out / f"{name}_progress.jsonl").open("x", buffering=1)

    def snap(cycle, index):
        torch.xpu.synchronize(device)
        m = libc.mallinfo2()
        row = dict(
            cycle=cycle,
            batch=index,
            step=record["completed_updates"],
            seconds=time.monotonic() - start,
            glibc_in_use=m.uordblks,
            glibc_free=m.fordblks,
            glibc_mmap=m.hblkhd,
            **telemetry(device),
        )
        record["snapshots"].append(row)
        progress.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def cycle(k):
        for i, lens in enumerate(selected, 1):
            if time.monotonic() - start >= 600:
                raise TimeoutError("Diagnostic time bound")
            record["active_cycle"] = k
            record["active_batch"] = i
            record["active_lengths"] = lens
            model.train()
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(4, max(lens), 64).to(device)
            mask = torch.ones_like(x, dtype=torch.bool)
            logits, lengths = model(x, mask, torch.tensor(lens))
            loss = ctc_objective(logits, lengths, targets)
            loss.backward()
            check_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            check_parameters_and_optimizer(model, optimizer)
            del x, mask, logits, lengths, loss
            record["completed_updates"] += 1
            if i % 32 == 0:
                snap(k, i)

    try:
        snap(0, 0)
        cycle(1)
        cycle(2)
        tracker = nullcontext()
        if args.profile:
            sys.path.append("/tmp/rclat_memray")
            import memray

            record["memray_version"] = memray.__version__
            tracker = memray.Tracker(
                out / f"{name}.bin",
                native_traces=True,
                file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS,
            )
        with tracker:
            cycle(3)
        record["stop_reason"] = "completed_three_passes"
    except Exception as exc:
        record.update(stop_reason="exception", error=repr(exc))
    record["seconds"] = time.monotonic() - start
    record["historical_primitive_failure_reproduced"] = (
        "could not create a primitive" in record.get("error", "")
    )
    (out / f"{name}_result.json").write_text(json.dumps(record, indent=2) + "\n")
    progress.close()
    print(json.dumps({k: v for k, v in record.items() if k != "snapshots"}), flush=True)


if __name__ == "__main__":
    main()
