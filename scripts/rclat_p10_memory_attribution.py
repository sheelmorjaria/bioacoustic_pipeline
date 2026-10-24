"""Bounded synthetic full-model memory attribution; never scientific scoring."""

import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import time
import tracemalloc

import torch
from bioacoustic_pipeline.representation.rclat_scale_v4 import ScaleCTC
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective
from bioacoustic_pipeline.representation.rclat_numerics import (
    check_gradients,
    check_parameters_and_optimizer,
)
from rclat_p10_c1_shape_replay import telemetry


class Mallinfo2(ctypes.Structure):
    _fields_ = [
        (n, ctypes.c_size_t)
        for n in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed", "varying"], required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    out = Path("outputs/rclat_p10_memory_attribution")
    out.mkdir(exist_ok=True)
    plan_path = out / f"{args.mode}_plan.json"
    if plan_path.exists():
        raise RuntimeError("Existing diagnostic plan; no retry")
    journal = Path("outputs/rclat_p10_corrected_v2/mel_steps.jsonl")
    rows = [json.loads(s) for s in journal.read_text().splitlines()]
    train = [r["native_lengths"] for r in rows if r["event"] == "forward_begin"][:120]
    validation = [r["native_lengths"] for r in rows if r["event"] == "validation_forward_begin"][:8]
    plan = dict(
        mode=args.mode,
        device=args.device,
        steps=120,
        wall_seconds=180,
        actual_model="ScaleCTC v4,64-input,width256,12layers,4heads,1024expansion,kernel15",
        source_journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        target="Synthetic40-character target; CPU CTC; no real transcripts or outcome scoring",
        training_lengths="Repeated first historical batch lengths"
        if args.mode == "fixed"
        else "First120historical native-length batches",
        validation="Same8historical validation-length batches after updates40,80,120;synthetic input,no decoding",
        optimizer="Fresh AdamW,lr1e-5,weight_decay0.01,clip5;diagnostic updates only",
        attribution="tracemalloc(5frames),glibc mallinfo2,smaps_rollup,RSS,XPU counters;no exact ownership claim across overlapping counters",
        interventions="At end only: snapshot; gc.collect; snapshot; malloc_trim(0); snapshot. No runtime-cache policy change.",
        limits="One process per condition on different XPUs;no device crossover,no historical full run,no source/cache changes,synthetic all-valid features.",
    )
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    libc = ctypes.CDLL(None)
    libc.mallinfo2.restype = Mallinfo2
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = f"xpu:{args.device}"
    torch.xpu.set_device(device)
    tracemalloc.start(5)
    snapshots = []
    progress = (out / f"{args.mode}_progress.jsonl").open("x", buffering=1)
    start = time.monotonic()

    def snap(phase, step):
        torch.xpu.synchronize(device)
        heap = libc.mallinfo2()
        pycurrent, pypeak = tracemalloc.get_traced_memory()
        row = dict(
            phase=phase,
            step=step,
            seconds=time.monotonic() - start,
            python_traced_current=pycurrent,
            python_traced_peak=pypeak,
            glibc_arena=heap.arena,
            glibc_in_use=heap.uordblks,
            glibc_free_in_arena=heap.fordblks,
            glibc_mmap=heap.hblkhd,
            **telemetry(device),
        )
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                if k in ("Rss", "Pss", "Anonymous", "Private_Dirty", "Private_Clean"):
                    row["smaps_" + k] = v.strip()
        snapshots.append(row)
        progress.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "phase",
                        "step",
                        "seconds",
                        "VmRSS",
                        "python_traced_current",
                        "glibc_in_use",
                        "glibc_mmap",
                        "xpu_reserved",
                    )
                }
            ),
            flush=True,
        )

    def batch(lengths):
        n = max(lengths)
        features = torch.randn(4, n, 64).to(device)
        valid = torch.ones(4, n, 64, dtype=torch.bool).to(device)
        return features, valid, torch.tensor(lengths)

    record = dict(
        completed_updates=0, scientific_training=False, scientific_validation_scoring=False
    )
    try:
        snap("before_model", 0)
        model = ScaleCTC(
            64, width=256, layers=12, heads=4, expansion=1024, kernel=15, dropout=0.0
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        targets = [torch.tensor(([2, 3, 4, 5] * 10), dtype=torch.long) for _ in range(4)]
        snap("initialized", 0)
        for step in range(1, 121):
            if time.monotonic() - start >= 180:
                record["stop_reason"] = "wall_time_bound"
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            inputs = batch(train[0] if args.mode == "fixed" else train[step - 1])
            selected = step in (1, 40, 80, 120)
            if selected:
                snap("before_forward", step)
            logits, lengths = model(*inputs)
            if selected:
                snap("after_forward", step)
            loss = ctc_objective(logits, lengths, targets)
            loss.backward()
            check_gradients(model)
            if selected:
                snap("after_backward", step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            check_parameters_and_optimizer(model, optimizer)
            del inputs, logits, lengths, loss
            record["completed_updates"] = step
            if step % 10 == 0 or selected:
                snap("after_update", step)
            if step % 40 == 0:
                model.eval()
                snap("validation_begin", step)
                with torch.no_grad():
                    for lens in validation:
                        inputs = batch(lens)
                        logits, lengths = model(*inputs)
                        if not bool(torch.isfinite(logits).all()):
                            raise FloatingPointError("Nonfinite synthetic eval logits")
                        del inputs, logits, lengths
                snap("validation_end", step)
        else:
            record["stop_reason"] = "completed120updates"
        snap("before_gc", record["completed_updates"])
        record["gc_collected"] = gc.collect()
        snap("after_gc", record["completed_updates"])
        record["malloc_trim_return"] = libc.malloc_trim(0)
        snap("after_malloc_trim", record["completed_updates"])
        record["python_top_live"] = [
            str(r) for r in tracemalloc.take_snapshot().statistics("lineno")[:15]
        ]
    except Exception as exc:
        record.update(stop_reason="exception", error=repr(exc))
        snap("failure", record["completed_updates"])
    record.update(
        seconds=time.monotonic() - start,
        snapshots=snapshots,
        historical_primitive_failure_reproduced="could not create a primitive"
        in record.get("error", ""),
    )
    (out / f"{args.mode}_result.json").write_text(json.dumps(record, indent=2) + "\n")
    progress.close()
    print(
        json.dumps({k: v for k, v in record.items() if k not in ["snapshots", "python_top_live"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
