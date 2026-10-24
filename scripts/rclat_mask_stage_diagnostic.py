"""Fresh-process, operation-boundary localisation of XPU mask/layout anomalies."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from bioacoustic_pipeline.diagnostics.rclat_mask_stages import worker_case, run_bounded, write

ROOT = Path("outputs/rclat_mask_stages_v1")
SOURCE = Path("outputs/rclat_rtpe_numerical_trace_v1/seed_101_case_0/checkpoints.pt")
CASES = [
    ("full_contiguous", "contiguous", "finite"),
    ("masked_inf_strided", "strided", "inf"),
    ("full_strided", "strided", "finite"),
    ("masked_nan_contiguous", "contiguous", "nan"),
    ("masked_nan_strided", "strided", "nan"),
    ("full_offset7", "offset7", "finite"),
]
OPERATIONS = ["transfer", "index", "and", "where"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def bindings():
    plan = json.loads((ROOT / "plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("Frozen dependency changed: " + path)
    return plan


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        SOURCE,
        Path("src/bioacoustic_pipeline/diagnostics/rclat_mask_stages.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_mask_operator.py"),
        Path("tests/test_rclat_mask_stages.py"),
        Path("outputs/rclat_mask_operator_v1/interrupted_closeout.json"),
        Path("outputs/rclat_mask_operator_v1/plan.json"),
    ]
    write(
        ROOT / "plan.json",
        dict(
            status="LOCALLY_FROZEN_STAGE_LOCALISATION_ONLY",
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            cases=[
                dict(name=name, layout=layout, variant=variant) for name, layout, variant in CASES
            ],
            operations=OPERATIONS,
            workers=24,
            worker_timeout_seconds=20,
            kill_cleanup_seconds=3,
            global_walltime_seconds=600,
            worker_isolation="Fresh subprocess for each case/operation;parent uses Popen timeout and kills only its own child;no reruns",
            device="xpu:0",
            precision="FP32",
            CPU_threads=2,
            requested_cache_capacity=16384,
            pipeline="CPU source snapshots;dense H2D+roundtrip checks;backing creation+checks;each layout copy+full backing check;pre-operation snapshots;single assigned operation;post-operation snapshots",
            preconditions="Abort worker before assigned operation on any source/transfer/layout mismatch;no fixing or contiguous fallback",
            journaling="Each primitive start/completion/failure fsynced;explicit synchronisation at operation boundaries;all roundtrip tensors retained",
            stop="Kill worker at20s;cleanup3s;stop global study if child unreaped or600s reached;never extend budget",
            scope="Fault-localisation,not full invalid/empty-support guard validation. Fresh contexts and synchronisation can change symptom;operator boundaries include synchronisation,not kernel internals.",
            model_forward_passes=0,
            optimizer_updates=0,
            production_guard_changed=False,
            corrected_replication_authorized=False,
            dependencies={str(p): sha(p) for p in sources},
        ),
    )
    print("24-worker staged diagnostic frozen; no device run.", flush=True)


def worker(case_index, operation):
    plan = bindings()
    if not 0 <= case_index < len(CASES) or operation not in OPERATIONS:
        raise ValueError("Unregistered worker")
    torch.set_num_threads(2)
    name, layout, variant = CASES[case_index]
    sources = torch.load(SOURCE, map_location="cpu", weights_only=True)
    values = sources["transfer.target"].clone()
    observed = sources["transfer.observed"].clone()
    if variant in ["nan", "inf"]:
        values.flatten()[0] = float(variant)
        observed.flatten()[0] = False
    result = worker_case(
        values,
        observed,
        layout,
        operation,
        ROOT / f"{case_index:02d}_{name}_{operation}",
        device=plan["device"],
    )
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


def last_stage(directory):
    file = directory / "stages.jsonl"
    if not file.exists():
        return dict(
            last_event=None, unfinished_stages=[], scope="Before durable worker stage journal"
        )
    rows = []
    for line in file.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    pending = []
    for row in rows:
        if row["event"] == "started":
            pending.append(row["stage"])
        elif row["event"] in ["completed", "failed"] and row["stage"] in pending:
            pending.remove(row["stage"])
    return dict(last_event=rows[-1] if rows else None, unfinished_stages=pending)


def run():
    plan = bindings()
    if (ROOT / "STARTED.json").exists():
        raise FileExistsError("Existing staged attempt preserved")
    if (
        os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384"
        or "DNNL_PRIMITIVE_CACHE_CAPACITY" in os.environ
    ):
        raise ValueError("Unchanged numerical/cache policy required")
    write(ROOT / "STARTED.json", dict(plan_sha256=sha(ROOT / "plan.json"), optimizer_updates=0))
    started = time.monotonic()
    rows = []
    stop = None
    for case_index, (name, _layout, _variant) in enumerate(CASES):
        for operation in OPERATIONS:
            if time.monotonic() - started >= plan["global_walltime_seconds"]:
                stop = "GLOBAL_WALLTIME"
                break
            bindings()
            directory = ROOT / f"{case_index:02d}_{name}_{operation}"
            write(
                ROOT / f"{case_index:02d}_{operation}_requested.json",
                dict(case_index=case_index, name=name, operation=operation),
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--case",
                str(case_index),
                "--operation",
                operation,
            ]
            process = run_bounded(
                command,
                ROOT / f"{case_index:02d}_{operation}_console.log",
                timeout=plan["worker_timeout_seconds"],
                cleanup=plan["kill_cleanup_seconds"],
            )
            row = dict(
                case=name,
                operation=operation,
                process=process,
                stage_progress=last_stage(directory),
            )
            if (directory / "result.json").exists():
                row["worker_result"] = json.loads((directory / "result.json").read_text())
            write(ROOT / f"{case_index:02d}_{operation}_supervisor.json", row)
            rows.append(row)
            status = (
                "TIMED_OUT"
                if process["timed_out"]
                else row.get("worker_result", {}).get("status", "EARLY_PROCESS_FAILURE")
            )
            print(
                f"{name}/{operation}: {status};unfinished={row['stage_progress']['unfinished_stages']}",
                flush=True,
            )
            if process["unreaped_after_kill"]:
                stop = "CHILD_NOT_REAPED_AFTER_KILL"
                break
        if stop:
            break
    bindings()
    result = dict(
        supervised_workers=len(rows),
        planned_workers=24,
        all_workers_supervised=len(rows) == 24,
        workers_with_result=sum("worker_result" in r for r in rows),
        timeouts=sum(r["process"]["timed_out"] for r in rows),
        statuses={
            s: sum(r.get("worker_result", {}).get("status") == s for r in rows)
            for s in [
                "PASS",
                "PRE_OPERATION_MISMATCH",
                "OPERATOR_MISMATCH",
                "POST_OPERATION_MISMATCH",
                "EXCEPTION",
            ]
        },
        elapsed_seconds=time.monotonic() - started,
        stop_reason=stop,
        workers=rows,
        model_forward_passes=0,
        scientific_optimizer_updates=0,
        guard_certified=False,
        production_guard_changed=False,
        replication_gate="UNEVALUATED",
        corrected_replication_authorized=False,
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    write(ROOT / "summary.json", result)
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.rglob("*") if p.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k != "workers"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "worker"])
    parser.add_argument("--case", type=int)
    parser.add_argument("--operation", choices=OPERATIONS)
    args = parser.parse_args()
    if args.action == "worker":
        sys.exit(worker(args.case, args.operation))
    {"prepare": prepare, "run": run}[args.action]()


if __name__ == "__main__":
    main()
