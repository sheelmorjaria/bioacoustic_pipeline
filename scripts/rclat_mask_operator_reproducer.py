"""GRU-free XPU mask-operator reproducer; no production guard replacement."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from bioacoustic_pipeline.diagnostics.rclat_mask_operator import (
    reference_selection,
    shape_preserving_checks,
    layout_copy,
)

ROOT = Path("outputs/rclat_mask_operator_v1")
CAPTURE_ROOT = Path("outputs/rclat_rtpe_numerical_trace_v1")
CAPTURES = [
    ("target_full", 101, 0, "transfer.target"),
    ("target307_partial", 307, 2, "transfer.target"),
    ("target401_partial", 401, 2, "transfer.target"),
    ("forecast307_partial", 307, 2, "forecast"),
]
PATTERNS = [
    "captured",
    "masked_nan",
    "masked_inf",
    "observed_nan",
    "observed_inf",
    "all_masked_nan",
    "empty_first_origin",
]
LAYOUTS = ["contiguous", "strided", "offset7"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def cases():
    result = []
    for name, seed, case, key in CAPTURES:
        p = CAPTURE_ROOT / f"seed_{seed}_case_{case}/checkpoints.pt"
        tensors = torch.load(p, map_location="cpu", weights_only=True)
        source = tensors[key]
        observed = tensors["transfer.observed"]
        for pattern in PATTERNS:
            a = source.clone()
            m = observed.clone()
            if pattern in ["masked_nan", "masked_inf"]:
                m.flatten()[0] = False
                a.flatten()[0] = float("nan") if pattern == "masked_nan" else float("inf")
            elif pattern in ["observed_nan", "observed_inf"]:
                position = int(torch.nonzero(m.flatten())[0])
                a.flatten()[position] = float("nan") if pattern == "observed_nan" else float("inf")
            elif pattern == "all_masked_nan":
                a.fill_(float("nan"))
                m.fill_(False)
            elif pattern == "empty_first_origin":
                a[0] = float("nan")
                m[0] = False
            for layout in LAYOUTS:
                result.append(
                    dict(
                        name=f"{name}.{pattern}.{layout}",
                        source=str(p),
                        pattern=pattern,
                        layout=layout,
                        x=a,
                        m=m,
                    )
                )
    for shape in [
        (1, 255),
        (1, 256),
        (1, 257),
        (1, 511),
        (1, 512),
        (1, 513),
        (1, 767),
        (1, 768),
        (1, 769),
        (4, 3, 64),
    ]:
        for partial in [False, True]:
            a = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
            m = torch.ones_like(a, dtype=torch.bool)
            if partial:
                m.flatten()[2] = False
                a.flatten()[2] = float("nan")
            result.append(
                dict(
                    name=f"control.{shape}.{partial}",
                    source="synthetic_arange",
                    pattern="one_masked_nan" if partial else "all_observed",
                    layout="contiguous",
                    x=a,
                    m=m,
                )
            )
    assert len(result) == 104
    return result


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    rows = cases()
    sources = {str(CAPTURE_ROOT / f"seed_{s}_case_{c}/checkpoints.pt") for _, s, c, _ in CAPTURES}
    sources.update(
        [
            str(Path(__file__)),
            "src/bioacoustic_pipeline/diagnostics/rclat_mask_operator.py",
            "tests/test_rclat_mask_operator.py",
            str(CAPTURE_ROOT / "localization_audit.json"),
        ]
    )
    schedule = [{k: v for k, v in r.items() if k not in ["x", "m"]} for r in rows]
    write(
        ROOT / "plan.json",
        dict(
            status="LOCALLY_FROZEN_OPERATOR_ONLY_DIAGNOSTIC",
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            cases=104,
            schedule=schedule,
            device="xpu:0",
            precision="FP32",
            requested_cache_capacity=16384,
            CPU_threads=2,
            walltime_seconds=180,
            primary="Native XPU values[observed] without GRU,hooks or dispatch tracing; compare exact cardinality and values against CPU Torch and independent NumPy",
            order="Construct destination layout,perform one native boolean gather,copy result to CPU,then inspect transfers and candidate guards",
            candidates="Diagnostic-only shape-preserving AND+explicitcount and WHERE+explicitcount validity predicates",
            reference="Require observed support in every origin;ignore masked NaN/Inf and reject observed NaN/Inf",
            limits="Finite matrix of shapes/layouts;no universal backend claim,no model-history/allocator-history equivalence guarantee",
            failure_policy="Each registered case once;durable case results;no replacement/retry or policy change",
            candidate_deployment=False,
            scientific_training=False,
            scientific_optimizer_updates=0,
            replication_restart_authorized=False,
            dependencies={p: sha(p) for p in sorted(sources)},
        ),
    )
    print("104-case operator-only plan locally frozen; no XPU run.", flush=True)


def bindings():
    p = json.loads((ROOT / "plan.json").read_text())
    if not all(sha(path) == digest for path, digest in p["dependencies"].items()):
        raise ValueError("Frozen dependency changed")
    return p


def run():
    plan = bindings()
    if (ROOT / "STARTED.json").exists():
        raise FileExistsError("Existing diagnostic attempt preserved")
    if (
        os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384"
        or "DNNL_PRIMITIVE_CACHE_CAPACITY" in os.environ
    ):
        raise ValueError("Unchanged registered cache policy required")
    if not torch.xpu.is_available():
        raise RuntimeError("XPU0 unavailable")
    write(ROOT / "STARTED.json", dict(plan_sha256=sha(ROOT / "plan.json"), optimizer_updates=0))
    torch.set_num_threads(2)
    started = time.monotonic()

    def alarm(*_):
        raise TimeoutError("Frozen180s operator diagnostic walltime")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(plan["walltime_seconds"])
    all_rows = cases()
    results = []
    for index, row in enumerate(all_rows):
        if time.monotonic() - started > plan["walltime_seconds"]:
            break
        directory = ROOT / f"case_{index:03d}"
        directory.mkdir(exist_ok=False)
        write(directory / "STARTED.json", dict(index=index, name=row["name"]))
        x, m = row["x"], row["m"]
        reference = reference_selection(x, m)
        expected = reference.pop("selected")
        cpu_selected = x[m].numpy()
        if not np.array_equal(cpu_selected, expected, equal_nan=True):
            raise ValueError("CPU Torch/NumPy reference disagrees")
        record = dict(
            index=index,
            name=row["name"],
            source=row["source"],
            pattern=row["pattern"],
            layout=row["layout"],
            reference=reference,
        )
        payload = dict(source=x.numpy(), observed=m.numpy(), CPU_selected=cpu_selected)
        try:
            dx, dm = layout_copy(x, m, row["layout"], "xpu:0")
            record.update(
                shape=list(dx.shape),
                value_stride=list(dx.stride()),
                mask_stride=list(dm.stride()),
                value_storage_offset=dx.storage_offset(),
                mask_storage_offset=dm.storage_offset(),
            )
            # Primary operation before any observation/reduction of the device tensors.
            try:
                selected = dx[dm]
                host = selected.detach().cpu().numpy()
                payload["XPU_selected"] = host
                record.update(
                    XPU_selected_count=int(host.size),
                    index_cardinality_equal=host.size == expected.size,
                    index_values_equal=bool(np.array_equal(host, expected, equal_nan=True)),
                    XPU_selected_nonfinite=int((~np.isfinite(host)).sum()),
                )
            except Exception:
                record["index_exception"] = traceback.format_exc()
                record["index_cardinality_equal"] = False
                record["index_values_equal"] = False
            source_after = dx.detach().cpu().numpy()
            mask_after = dm.detach().cpu().numpy()
            record["source_values_unchanged"] = bool(
                np.array_equal(source_after, x.numpy(), equal_nan=True)
            )
            record["mask_values_unchanged"] = bool(np.array_equal(mask_after, m.numpy()))
            payload["source_after_gather"] = source_after
            payload["mask_after_gather"] = mask_after
            checks = shape_preserving_checks(dx, dm)
            record["candidate_checks"] = checks
            record["candidate_AND_matches_reference"] = checks["and_valid"] == reference["valid"]
            record["candidate_WHERE_matches_reference"] = (
                checks["where_valid"] == reference["valid"]
            )
            record["candidate_counts_match_reference"] = (
                checks["observed_counts"] == reference["observed_counts"]
                and checks["observed_nonfinite_counts"] == reference["observed_nonfinite_counts"]
            )
        except Exception:
            record["case_exception"] = traceback.format_exc()
        with (directory / "tensors.npz").open("xb") as f:
            np.savez_compressed(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        write(directory / "result.json", record)
        results.append(record)
        if (index + 1) % 20 == 0:
            print(f"Completed{index + 1}/104operator cases", flush=True)
    signal.alarm(0)
    bindings()
    candidates_ok = all(
        r.get("candidate_AND_matches_reference")
        and r.get("candidate_WHERE_matches_reference")
        and r.get("candidate_counts_match_reference")
        for r in results
    )
    summary = dict(
        completed_cases=len(results),
        planned_cases=104,
        completed=len(results) == 104,
        indexing_cardinality_mismatches=sum(
            not r.get("index_cardinality_equal", False) for r in results
        ),
        indexing_value_mismatches=sum(not r.get("index_values_equal", False) for r in results),
        candidate_checks_all_match_CPU=candidates_ok,
        source_or_mask_changed_cases=sum(
            not r.get("source_values_unchanged", False) or not r.get("mask_values_unchanged", False)
            for r in results
        ),
        elapsed_seconds=time.monotonic() - started,
        scientific_optimizer_updates=0,
        model_forward_passes=0,
        production_guard_changed=False,
        replication_restart_authorized=False,
        replication_gate="UNEVALUATED",
        original_RTPE1_outcome="Closed PASS unchanged",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    write(ROOT / "summary.json", summary)
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.rglob("*") if p.is_file()})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.action]()


if __name__ == "__main__":
    main()
