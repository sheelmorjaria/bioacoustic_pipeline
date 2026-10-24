"""Frozen missing-data guard semantics matrix; diagnostic only, no deployment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from bioacoustic_pipeline.diagnostics.rclat_guard_matrix import guard_fixture, schedule
from bioacoustic_pipeline.diagnostics.rclat_mask_operator import reference_selection
from bioacoustic_pipeline.diagnostics.rclat_mask_stages import worker_case, run_bounded, write

ROOT = Path("outputs/rclat_guard_validation_v1")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def bindings():
    plan = json.loads((ROOT / "plan.json").read_text())
    if not all(sha(p) == h for p, h in plan["dependencies"].items()):
        raise ValueError("Frozen guard validation dependency changed")
    return plan


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_guard_matrix.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_mask_stages.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_mask_operator.py"),
        Path("tests/test_rclat_guard_matrix.py"),
        Path("outputs/rclat_mask_stages_v1/localization_audit.json"),
    ]
    write(
        ROOT / "plan.json",
        dict(
            status="LOCALLY_FROZEN_GUARD_SEMANTICS_VALIDATION_ONLY",
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            workers=schedule(),
            worker_count=132,
            worker_timeout_seconds=20,
            kill_cleanup_seconds=3,
            global_active_walltime_seconds=600,
            shapes=[[4, 3, 64], [128, 3, 64]],
            layouts=["contiguous", "strided", "offset7"],
            predicates=["and", "where"],
            meaning="Each origin must have observed support;observed NaN/+Inf/-Inf reject;unsupported nonfinite values ignored;counts must match NumPy",
            source="Deterministic synthetic arangeFP32/1024 values and deliberately placed nonfinites;static independently declared expected verdicts",
            device="xpu:0",
            precision="FP32",
            CPU_threads=2,
            requested_cache_capacity=16384,
            checks="Exact source/dense transfer/backing layout/pre/post snapshots,per-origin observed counts,AND per-origin nonfinite counts,and reference verdicts",
            isolation="One fresh process per pattern/shape/layout/predicate;no native boolean selection;each primitive start/completion/failure fsynced",
            scope="Certification only of diagnostic predicates over registered finite matrix with verified inputs,explicit fences and retained transfer tensors;not original unobserved lifetime/history behaviour,arbitrary shapes or production correction",
            acceptance="For each predicate:all66registered workers finish under bound,pre/post snapshots exact,reference verdict correct,observed counts correct,and AND bad counts correct. Any missing/timeout/error/mismatch prevents that predicate certificate.",
            no_retry=True,
            scientific_optimizer_updates=0,
            model_forward_passes=0,
            production_guard_changed=False,
            replication_restart_authorized=False,
            dependencies={str(p): sha(p) for p in paths},
        ),
    )
    print("132-worker guard semantics plan frozen; no XPU run.", flush=True)


def worker(index):
    plan = bindings()
    if not 0 <= index < len(plan["workers"]):
        raise ValueError("Unregistered job")
    row = plan["workers"][index]
    torch.set_num_threads(2)
    x, m = guard_fixture(tuple(row["shape"]), row["pattern"])
    reference = reference_selection(x, m)
    if reference["valid"] != row["expected_valid"]:
        raise ValueError("Independent reference/static fixture labels disagree")
    result = worker_case(
        x, m, row["layout"], row["operation"], ROOT / f"worker_{index:03d}", device=plan["device"]
    )
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


def stage_progress(directory):
    path = directory / "stages.jsonl"
    if not path.exists():
        return dict(last_event=None, unfinished_stages=[], scope="Before worker journal")
    rows = []
    for line in path.read_text().splitlines():
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
        raise FileExistsError("Existing validation attempt preserved;no retry")
    if (
        os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384"
        or "DNNL_PRIMITIVE_CACHE_CAPACITY" in os.environ
    ):
        raise ValueError("Unchanged numerical policy required")
    write(ROOT / "STARTED.json", dict(plan_sha256=sha(ROOT / "plan.json"), optimizer_updates=0))
    start = time.monotonic()
    results = []
    stop = None
    for index, row in enumerate(plan["workers"]):
        remaining = plan["global_active_walltime_seconds"] - (time.monotonic() - start)
        if remaining <= 0:
            stop = "GLOBAL_WALLTIME"
            break
        bindings()
        write(ROOT / f"worker_{index:03d}_request.json", row)
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--index", str(index)]
        process = run_bounded(
            command,
            ROOT / f"worker_{index:03d}_console.log",
            timeout=min(remaining, plan["worker_timeout_seconds"]),
            cleanup=plan["kill_cleanup_seconds"],
        )
        directory = ROOT / f"worker_{index:03d}"
        record = dict(
            index=index, fixture=row, process=process, stage_progress=stage_progress(directory)
        )
        if (directory / "result.json").exists():
            record["worker_result"] = json.loads((directory / "result.json").read_text())
        write(ROOT / f"worker_{index:03d}_supervisor.json", record)
        results.append(record)
        if (
            (index + 1) % 12 == 0
            or process["timed_out"]
            or record.get("worker_result", {}).get("status") != "PASS"
        ):
            print(
                f"Guard validation {index + 1}/132;last={row['name']};status={record.get('worker_result', {}).get('status', 'TIMED_OUT_OR_NO_RESULT')}",
                flush=True,
            )
        if process["unreaped_after_kill"]:
            stop = "CHILD_NOT_REAPED_AFTER_KILL"
            break
    bindings()
    certificates = {}
    for predicate in plan["predicates"]:
        rows = [r for r in results if r["fixture"]["operation"] == predicate]
        passed = sum(
            not r["process"]["timed_out"]
            and r["process"]["returncode"] == 0
            and r.get("worker_result", {}).get("status") == "PASS"
            and r["worker_result"]["input_roundtrip_verified"]
            for r in rows
        )
        certificates[predicate] = dict(
            planned_workers=66,
            recorded_workers=len(rows),
            passed_workers=passed,
            registered_matrix_passed=passed == 66,
        )
    result = dict(
        completed_workers=len(results),
        planned_workers=132,
        all_workers_recorded=len(results) == 132,
        certificates=certificates,
        verdict_reject_workers=sum(not r["fixture"]["expected_valid"] for r in results),
        verdict_accept_workers=sum(r["fixture"]["expected_valid"] for r in results),
        timeouts=sum(r["process"]["timed_out"] for r in results),
        elapsed_seconds=time.monotonic() - start,
        stop_reason=stop,
        predicates_registered_matrix_passed=all(
            r["registered_matrix_passed"] for r in certificates.values()
        ),
        certification_scope=plan["scope"],
        model_forward_passes=0,
        scientific_optimizer_updates=0,
        production_guard_changed=False,
        corrected_replication_authorized=False,
        replication_gate="UNEVALUATED",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
        worker_records=results,
    )
    write(ROOT / "summary.json", result)
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.rglob("*") if p.is_file()})
    print(
        json.dumps({k: v for k, v in result.items() if k != "worker_records"}, indent=2), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "worker"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        sys.exit(worker(args.index))
    {"prepare": prepare, "run": run}[args.action]()


if __name__ == "__main__":
    main()
