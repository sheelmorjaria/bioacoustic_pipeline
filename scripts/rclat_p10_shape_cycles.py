"""Finite-shape cycling/device crossover and optional native allocation stacks."""

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
    parser.add_argument("--mode", choices=["fixed", "cycling"], required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    out = Path("outputs/rclat_p10_shape_cycles")
    out.mkdir(exist_ok=True)
    name = f"{args.mode}_xpu{args.device}" + ("_profiled" if args.profile else "")
    planpath = out / f"{name}_plan.json"
    if planpath.exists():
        raise RuntimeError("Existing attempt; no automatic retry")
    journal = Path("outputs/rclat_p10_corrected_v2/mel_steps.jsonl")
    unique = []
    seen = set()
    for line in journal.open():
        row = json.loads(line)
        if row["event"] != "forward_begin":
            continue
        shape = (max(row["native_lengths"]) + 3) // 4
        if shape not in seen:
            unique.append(row["native_lengths"])
            seen.add(shape)
        if len(unique) == 12:
            break
    assert len(unique) == 12
    plan = dict(
        mode=args.mode,
        device=args.device,
        profile=args.profile,
        cycles=5,
        batches_per_cycle=12,
        steps=60,
        seconds_bound=300,
        lengths=unique if args.mode == "cycling" else [unique[0]] * 12,
        source_journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        precision="FP32,deterministic",
        architecture="v4ScaleCTC64input,width256,12layers,4heads,expansion1024,kernel15",
        inputs="Synthetic CPU-generated features transferred to assigned XPU;all-valid masks;synthetic40-character CTC target",
        optimizer="Fresh AdamW lr1e-5,wd0.01;clip5;no scientific checkpoint",
        profiler="Memray1.20 native aggregated allocations,cycles2..5 only;cycle1warmup unprofiled"
        if args.profile
        else "none",
        endpoint="Memory/stacks only;no transcript decoding or validation outcome",
        limits="60updates;no checkpoint/dataIO/validation transitions;no long-run stability claim;independent fresh process per condition",
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

    def snap(cycle):
        torch.xpu.synchronize(device)
        m = libc.mallinfo2()
        row = dict(
            cycle=cycle,
            seconds=time.monotonic() - start,
            glibc_in_use=m.uordblks,
            glibc_free=m.fordblks,
            glibc_mmap=m.hblkhd,
            known_parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
            known_optimizer_bytes=sum(
                v.numel() * v.element_size()
                for s in optimizer.state.values()
                for v in s.values()
                if torch.is_tensor(v)
            ),
            **telemetry(device),
        )
        record["snapshots"].append(row)
        progress.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def cycle(k):
        for lengths in plan["lengths"]:
            if time.monotonic() - start >= 300:
                raise TimeoutError("Diagnostic time bound")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(4, max(lengths), 64).to(device)
            masks = torch.ones_like(x, dtype=torch.bool)
            logits, lens = model(x, masks, torch.tensor(lengths))
            loss = ctc_objective(logits, lens, targets)
            loss.backward()
            check_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            check_parameters_and_optimizer(model, optimizer)
            del x, masks, logits, lens, loss
            record["completed_updates"] += 1
        snap(k)

    try:
        snap(0)
        cycle(1)
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
            for k in range(2, 6):
                cycle(k)
        record["stop_reason"] = "completed_five_cycles"
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
