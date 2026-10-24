"""Preflight-only per-case fault localisation; preserves frozen model/loss operators."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from bioacoustic_pipeline.diagnostics.rclat_rtpe_trace import (
    CaseRecorder,
    NumericalTrace,
    attach_checkpoint_hooks,
)
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write
from bioacoustic_pipeline.representation.rclat_rtpe_pool import PoolTemporalPredictionEngine
from bioacoustic_pipeline.representation.rclat_rtpe_weighted import minibatch_mse

ROOT = Path("outputs/rclat_rtpe_numerical_trace_v1")
PAIRS = [
    (101, 2026102311),
    (211, 2026102312),
    (307, 2026102313),
    (401, 2026102314),
    (503, 2026102315),
]
FIXTURE = Path("outputs/rclat_rtpe1_protocol_v2/training_numerical_fixtures.npz")


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    original = load_script("original_replication", "scripts/rclat_rtpe_replication.py")
    original.bindings()
    assert (original.ROOT / "FAILURE.json").exists() and not (
        original.ROOT / "RUN_REQUESTED.json"
    ).exists()
    ROOT.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        FIXTURE,
        Path("scripts/rclat_rtpe_replication.py"),
        Path("outputs/rclat_rtpe_multiseed_v1/protocol.json"),
        Path("outputs/rclat_rtpe_multiseed_v1/execution_manifest.json"),
        Path("outputs/rclat_rtpe_multiseed_v1/FAILURE.json"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_trace.py"),
        Path("tests/test_rclat_rtpe_trace.py"),
        Path("scripts/rclat_rtpe_protocol_prepare_v2.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_pool.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_weighted.py"),
    ]
    write(
        ROOT / "plan.json",
        dict(
            status="LOCALLY_FROZEN_PREFLIGHT_ONLY_DIAGNOSTIC",
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            seed_pairs=PAIRS,
            cases=[0, 1, 2],
            device="xpu:0",
            order="same seed-major case-major order as failed preflight;each case once",
            numerical_policy="FP32;twoCPUthreads;requestedcache16384;original model,weights,fixtures and loss unchanged",
            relative_tolerance=1e-4,
            walltime_seconds=300,
            scientific_optimizer_updates=0,
            evaluation_scoring=False,
            observations=[
                "CPUfixture and observed masks",
                "exact parameter/input/target/mask transfer values",
                "masked multiscale encoder input",
                "GRU returned sequence and hidden state",
                "forecast-head input/output",
                "original weighted-loss dispatched operators and scalar",
                "parameter/input gradients",
            ],
            durability="Each case start and final record exclusive;JSONL events flushed;case completion/failure fsynced before nextcase;firstnonfinite snapshot and activations retained",
            failure_policy="Continue remaining independently specified cases after local numerical failure;no case retry,no operator replacement,no corrective replication run",
            interpretation="Tracing/copies synchronise devices and may alter timing/dispatch. Passing instrumented cases do not certify correction or authorise replication. First nonfinite returned tensor is not proof of internal allocation/kernel cause;ignore uninitialised allocation outputs.",
            dependencies={str(p): sha(p) for p in sources},
        ),
    )
    print("Trace diagnostic frozen; no numerical run yet.", flush=True)


def bindings():
    plan = json.loads((ROOT / "plan.json").read_text())
    for p, h in plan["dependencies"].items():
        if sha(p) != h:
            raise ValueError("Changed diagnostic dependency: " + p)
    return plan


def case_run(seed, case, data, helper, recorder, *, device="xpu:0"):
    payload = {}
    checkpoint_records = []
    handles = []

    def checkpoint(name, tensor, observed=None):
        info = recorder.tensor("checkpoint." + name, tensor, observed=observed)
        checkpoint_records.append(info)
        payload[name] = tensor.detach().cpu().clone()
        return info

    def transferred(name, source, device_value, observed=None):
        checkpoint("source." + name, source, observed)
        info = checkpoint("transfer." + name, device_value, observed)
        host = device_value.detach().cpu()
        finite = torch.isfinite(source) & torch.isfinite(host)
        equal_nonfinite = torch.equal(torch.isnan(source), torch.isnan(host)) and torch.equal(
            torch.isinf(source), torch.isinf(host)
        )
        recorder.event(
            name="transfer.comparison." + name,
            exact_finite_values_equal=bool(torch.equal(source[finite], host[finite])),
            nonfinite_locations_equal=bool(equal_nonfinite),
            observed_nonfinite=info["observed_nonfinite"],
        )

    result = {}
    status = "EXCEPTION"
    try:
        torch.manual_seed(seed)
        cpu = PoolTemporalPredictionEngine(64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16))
        gpu = copy.deepcopy(cpu).to(device)
        x = torch.from_numpy(data[f"x_{case}"])
        v = torch.from_numpy(data[f"valid_{case}"])
        y = torch.from_numpy(data[f"y_{case}"])
        observed = torch.from_numpy(data[f"observed_{case}"])
        weights = torch.from_numpy(data["population_weights"])
        indices = torch.from_numpy(data[f"indices_{case}"])
        reference = helper.run_grad(cpu, x, v, y, observed, weights, indices)
        for name, tensor in zip(
            ["forecast", "loss", "parameter_gradients", "input_gradients"], reference
        ):
            checkpoint("CPUreference." + name, tensor)
        dx = x.to(device)
        dv = v.to(device)
        dy = y.to(device)
        dm = observed.to(device)
        transferred("input", x, dx, v)
        transferred("valid", v, dv)
        transferred("target", y, dy, observed)
        transferred("observed", observed, dm)
        for name, param in gpu.named_parameters():
            transferred("parameter." + name, cpu.state_dict()[name], param.detach())
        checkpoint("population_weights", weights)
        handles = attach_checkpoint_hooks(gpu, checkpoint)
        gpu.zero_grad(set_to_none=True)
        dx = dx.detach().requires_grad_()
        with NumericalTrace(recorder, "forward"):
            forecast = gpu(dx, dv)["forecast"][:, -1]
        checkpoint("forecast", forecast)
        # Host-only observed views avoid warming the device boolean-index kernel before the unchanged guard.
        checkpoint("host_observed_forecast", forecast.detach().cpu()[observed])
        checkpoint("host_observed_target", dy.detach().cpu()[observed])
        with NumericalTrace(recorder, "weighted_loss"):
            loss = minibatch_mse(forecast, dy, dm, weights, indices)
        checkpoint("weighted_loss", loss)
        with NumericalTrace(recorder, "backward"):
            loss.backward()
        grads = []
        for name, param in gpu.named_parameters():
            if param.grad is not None:
                checkpoint("gradient." + name, param.grad)
                grads.append(param.grad.detach().cpu().flatten())
        checkpoint("input_gradient", dx.grad)
        actual = [
            forecast.detach().cpu(),
            loss.detach().cpu().reshape(1),
            torch.cat(grads),
            dx.grad.detach().cpu(),
        ]
        metrics = {}
        for name, a, b in zip(
            ["forecast", "loss", "global_parameter_gradients", "input_gradients"], reference, actual
        ):
            metrics[name] = (
                None
                if not torch.isfinite(b).all()
                else float((a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-30))
            )
        result["relative_errors"] = metrics
        status = (
            "PASSED_INSTRUMENTED"
            if all(e is not None and e < 1e-4 for e in metrics.values())
            else "NUMERICAL_FAILURE"
        )
    except Exception:
        result["traceback"] = traceback.format_exc()
        recorder.event(name="case.exception", traceback=result["traceback"])
    finally:
        for handle in handles:
            handle.remove()
        if payload:
            with (recorder.root / "checkpoints.pt").open("xb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
    result["checkpoint_summaries"] = checkpoint_records
    result["first_nonfinite_checkpoint"] = next(
        (
            info
            for info in checkpoint_records
            if info["observed_nonfinite"] or info["finiteness_flag_disagreements"]
        ),
        None,
    )
    return status, result


def run():
    plan = bindings()
    if (ROOT / "STARTED.json").exists():
        raise FileExistsError("No retry over an existing diagnostic attempt")
    if (
        os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384"
        or "DNNL_PRIMITIVE_CACHE_CAPACITY" in os.environ
    ):
        raise ValueError("Unchanged registered cache policy required")
    if not torch.xpu.is_available():
        raise RuntimeError("XPU0 unavailable")
    write(ROOT / "STARTED.json", dict(plan_sha256=sha(ROOT / "plan.json"), optimizer_updates=0))
    helper = load_script("numerics", "scripts/rclat_rtpe_protocol_prepare_v2.py")
    with np.load(FIXTURE, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    torch.set_num_threads(2)
    started = time.monotonic()
    rows = []

    def alarm(*_):
        raise TimeoutError("Frozen300s diagnostic walltime")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(300)
    for seed, _ in PAIRS:
        for case in plan["cases"]:
            if time.monotonic() - started > plan["walltime_seconds"]:
                break
            recorder = CaseRecorder(ROOT / f"seed_{seed}_case_{case}", seed=seed, case=case)
            status, result = case_run(seed, case, data, helper, recorder)
            row = recorder.finish(status, result)
            rows.append(row)
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        case=case,
                        status=status,
                        first_unexpected_nonfinite=row["first_unexpected_nonfinite"],
                    )
                ),
                flush=True,
            )
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
    signal.alarm(0)
    bindings()
    summary = dict(
        completed_cases=len(rows),
        planned_cases=15,
        cases=rows,
        elapsed_seconds=time.monotonic() - started,
        instrumented_cases_passed=sum(r["status"] == "PASSED_INSTRUMENTED" for r in rows),
        cases_with_nonfinite_capture=sum(r["first_unexpected_nonfinite"] is not None for r in rows),
        optimizer_updates=0,
        scientific_training_started=False,
        internal_evaluation_scoring=False,
        original_replication_gate="UNEVALUATED",
        corrected_replication_authorized=False,
        observation_limit="Tensor copies/finiteness observations synchronise and may alter symptom;finite instrumented runs are not correction certification",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    write(ROOT / "summary.json", summary)
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.rglob("*") if p.is_file()})
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.action]()


if __name__ == "__main__":
    main()
