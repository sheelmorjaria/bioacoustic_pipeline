"""Guard-only candidate numerical integration; never starts scientific training."""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from bioacoustic_pipeline.diagnostics.rclat_mask_stages import run_bounded
from bioacoustic_pipeline.diagnostics.rclat_guarded_validation import fault_schedule, fault_fixture
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write
from bioacoustic_pipeline.representation.rclat_rtpe_pool import PoolTemporalPredictionEngine
from bioacoustic_pipeline.representation.rclat_rtpe_weighted import minibatch_mse
from bioacoustic_pipeline.representation.rclat_rtpe_guarded import (
    GuardedPoolTemporalPredictionEngine,
    guarded_minibatch_mse,
)

ROOT = Path("outputs/rclat_rtpe_guard_integration_xpu1_v1")
FIXTURE = Path("outputs/rclat_rtpe1_protocol_v2/training_numerical_fixtures.npz")
SEEDS = [101, 211, 307, 401, 503]


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    certificate = Path("outputs/rclat_guard_validation_v1/certificate.json")
    assert (
        json.loads(certificate.read_text())["status"]
        == "BOTH_DIAGNOSTIC_PREDICATES_PASSED_REGISTERED_MATRIX"
    )
    paths = [
        Path(__file__),
        certificate,
        FIXTURE,
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_guarded.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_pool.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_weighted.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_guarded_validation.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_mask_stages.py"),
        Path("tests/test_rclat_rtpe_guarded.py"),
        Path("tests/test_rclat_guarded_validation.py"),
    ]
    write(
        ROOT / "plan.json",
        dict(
            status="LOCALLY_FROZEN_GUARD_ONLY_ENGINEERING_INTEGRATION",
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            chosen_predicate="WHERE finite check with existing support contract;chosen before integration outcomes",
            changes="Only input and weighted-loss observed-cell finite guards in a new candidate module;original frozen sources/entrypoints unchanged",
            unchanged="State initialisation,architecture,pooling,forecast readout,feature masks,weighted residual/square/mean expressions,weight normalisation,minibatch estimator",
            reference="Frozen CPU original model/loss;newCPU same state;newXPU same state;independent NumPy loss and analytic logit gradient",
            seeds=SEEDS,
            real_fixture_cases=[0, 1, 2],
            synthetic_full_batch=128,
            CPU_old_new_relative_tolerance=1e-7,
            CPU_XPU_relative_tolerance=1e-4,
            analytic_relative_tolerance=1e-6,
            metrics=[
                "forecast",
                "loss",
                "global_parameter_gradients",
                "input_gradients",
                "logit_gradients",
            ],
            forward_identity="AST-equivalent forward arithmetic except renamed guard call;identical state keys and tensors",
            fault_schedule=fault_schedule(),
            fault_cases=288,
            fault_positions="First,middle,last flattened positions in prediction,target and input;input final empty frame;first/all empty target origins",
            layouts=["contiguous", "strided", "offset7"],
            fault_batches=[4, 128],
            precision="FP32",
            device="xpu:1",
            CPU_threads=2,
            requested_cache_capacity=16384,
            history="Same-process seed-major numerical fixtures without dispatch tracing;fault inputs retain transfer buffers and verify layout copies before calls",
            worker_timeout_seconds=300,
            kill_cleanup_seconds=3,
            failure_policy="Each planned case once;durable before/after records;any exception,parity failure,wrong validity or timeout prevents integration certificate;no scientific retry",
            production_adoption=False,
            scientific_optimizer_updates=0,
            scientific_evaluation_scores=0,
            replication_restart_authorized=False,
            dependencies={str(p): sha(p) for p in paths},
        ),
    )
    print("Guard-only numerical integration protocol frozen; no scientific run.", flush=True)


def bindings():
    p = json.loads((ROOT / "plan.json").read_text())
    if not all(sha(path) == digest for path, digest in p["dependencies"].items()):
        raise ValueError("Frozen integration dependency changed")
    return p


def relative(a, b):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite parity tensor")
    return float((a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-30))


def compute(model, loss_function, x, valid, y, observed, weights, indices):
    model.zero_grad(set_to_none=True)
    leaf = x.detach().requires_grad_()
    p = model(leaf, valid)["forecast"][:, -1]
    p.retain_grad()
    loss = loss_function(p, y, observed, weights, indices)
    loss.backward()
    g = torch.cat(
        [v.grad.detach().cpu().flatten() for v in model.parameters() if v.grad is not None]
    )
    return [
        p.detach().cpu(),
        loss.detach().cpu().reshape(1),
        g,
        leaf.grad.detach().cpu(),
        p.grad.detach().cpu(),
    ]


def analytic(prediction, target, mask, selected_weights, factor):
    p = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    residual = np.zeros_like(p)
    np.subtract(p, y, out=residual, where=m)
    counts = m.reshape(len(m), -1).sum(1)
    loss = factor * np.sum(selected_weights * (residual**2).reshape(len(m), -1).sum(1) / counts)
    gradient = factor * 2 * selected_weights[:, None, None] * residual / counts[:, None, None]
    return loss, gradient


def numerical_case(name, seed, x, v, y, m, w, indices, plan):
    directory = ROOT / name
    directory.mkdir(exist_ok=False)
    write(directory / "STARTED.json", dict(seed=seed, input_shape=list(x.shape)))
    try:
        torch.manual_seed(seed)
        original = PoolTemporalPredictionEngine(
            64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16)
        )
        cpu = GuardedPoolTemporalPredictionEngine(
            64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16)
        )
        cpu.load_state_dict(original.state_dict())
        gpu = copy.deepcopy(cpu).to("xpu:1")
        assert all(
            torch.equal(original.state_dict()[k], cpu.state_dict()[k])
            for k in original.state_dict()
        )
        a = compute(original, minibatch_mse, x, v, y, m, w, indices)
        b = compute(cpu, guarded_minibatch_mse, x, v, y, m, w, indices)
        gx = x.to("xpu:1")
        gv = v.to("xpu:1")
        gy = y.to("xpu:1")
        gm = m.to("xpu:1")
        c = compute(gpu, guarded_minibatch_mse, gx, gv, gy, gm, w, indices)
        cpu_errors = {key: relative(left, right) for key, left, right in zip(plan["metrics"], a, b)}
        device_errors = {
            key: relative(left, right) for key, left, right in zip(plan["metrics"], a, c)
        }
        target_equal = np.array_equal(gy.detach().cpu().numpy(), y.numpy(), equal_nan=True)
        masks_equal = torch.equal(gv.cpu(), v) and torch.equal(gm.cpu(), m)
        expected_loss, expected_gradient = analytic(
            c[0].numpy(), y.numpy(), m.numpy(), w.numpy()[indices.numpy()], len(w) / len(indices)
        )
        analytic_loss = abs(float(c[1]) - expected_loss) / max(abs(expected_loss), 1e-30)
        analytic_gradient = relative(torch.from_numpy(expected_gradient), c[4])
        passed = (
            max(cpu_errors.values()) < plan["CPU_old_new_relative_tolerance"]
            and max(device_errors.values()) < plan["CPU_XPU_relative_tolerance"]
            and max(analytic_loss, analytic_gradient) < plan["analytic_relative_tolerance"]
            and target_equal
            and masks_equal
        )
        result = dict(
            passed=passed,
            seed=seed,
            CPU_old_new_errors=cpu_errors,
            CPU_XPU_errors=device_errors,
            analytic_loss_relative_error=analytic_loss,
            analytic_logit_gradient_relative_error=analytic_gradient,
            transferred_target_exact=bool(target_equal),
            transferred_masks_exact=bool(masks_equal),
        )
        with (directory / "parity_tensors.pt").open("xb") as f:
            torch.save(dict(original_CPU=a, candidate_CPU=b, candidate_XPU=c), f)
    except Exception:
        result = dict(passed=False, seed=seed, traceback=traceback.format_exc())
    write(directory / "result.json", result)
    print(name, "PASS" if result["passed"] else "FAIL", flush=True)
    return result


def pack(values, mask, layout):
    dense = values.to("xpu:1")
    dm = mask.to("xpu:1")
    keep = [dense, dm]
    if layout == "contiguous":
        a, b = dense, dm
    else:
        shape = (
            (*values.shape[:-1], values.shape[-1] * 2)
            if layout == "strided"
            else (values.numel() + 7,)
        )
        base = torch.full(shape, 17.0, dtype=values.dtype, device="xpu:1")
        mbase = torch.zeros(shape, dtype=torch.bool, device="xpu:1")
        keep.extend([base, mbase])
        if layout == "strided":
            a, b = base[..., ::2], mbase[..., ::2]
        else:
            a, b = base[7:].reshape(values.shape), mbase[7:].reshape(values.shape)
        a.copy_(dense)
        torch.xpu.synchronize()
        b.copy_(dm)
        torch.xpu.synchronize()
    if not np.array_equal(
        a.detach().cpu().numpy(), values.numpy(), equal_nan=True
    ) or not torch.equal(b.cpu(), mask):
        raise ValueError("Fault-fixture transfer/layout precondition failed")
    return a, b, keep


def fault_case(index, row, model, plan):
    directory = ROOT / f"fault_{index:03d}"
    directory.mkdir(exist_ok=False)
    write(directory / "STARTED.json", row)
    try:
        data = fault_fixture(row)
        if row["family"] == "input":
            x, mask, keep = pack(data["values"], data["mask"], row["layout"])
            try:
                with torch.no_grad():
                    forecast = model(x, mask)["forecast"]
                accepted = True
                finite = bool(torch.isfinite(forecast).all().cpu())
                error = None
            except ValueError as e:
                accepted = False
                finite = True
                error = str(e)
            expected_role = "supported" if row["kind"] == "empty_frame" else "features"
            passed = (
                accepted == row["expected_accept"]
                and finite
                and (accepted or expected_role in error)
            )
            unchanged = np.array_equal(
                x.detach().cpu().numpy(), data["values"].numpy(), equal_nan=True
            ) and torch.equal(mask.cpu(), data["mask"])
            result = dict(
                passed=bool(passed and unchanged),
                accepted=accepted,
                error=error,
                post_inputs_unchanged=bool(unchanged),
            )
        else:
            p, mask, keep1 = pack(data["prediction"], data["mask"], row["layout"])
            y, other_mask, keep2 = pack(data["target"], data["mask"], row["layout"])
            assert torch.equal(mask.cpu(), other_mask.cpu())
            p = p.detach().requires_grad_()
            w = torch.full((row["batch"],), 1 / row["batch"], dtype=torch.float64)
            indices = torch.arange(row["batch"])
            try:
                loss = guarded_minibatch_mse(p, y, mask, w, indices)
                loss.backward()
                accepted = True
                error = None
                expected_loss, expected_gradient = analytic(
                    data["prediction"].numpy(),
                    data["target"].numpy(),
                    data["mask"].numpy(),
                    w.numpy(),
                    1.0,
                )
                loss_error = abs(float(loss.detach().cpu()) - expected_loss) / max(
                    abs(expected_loss), 1e-30
                )
                gradient_error = relative(
                    torch.from_numpy(expected_gradient), p.grad.detach().cpu()
                )
                finite = max(loss_error, gradient_error) < plan["analytic_relative_tolerance"]
            except ValueError as e:
                accepted = False
                error = str(e)
                finite = True
                loss_error = None
                gradient_error = None
            expected_role = {
                "prediction": "predictions",
                "target": "targets",
                "support": "Every origin",
                "masked_pair": "unused",
            }[row["role"]]
            passed = (
                accepted == row["expected_accept"]
                and finite
                and (accepted or expected_role in error)
            )
            unchanged = (
                np.array_equal(p.detach().cpu().numpy(), data["prediction"].numpy(), equal_nan=True)
                and np.array_equal(y.detach().cpu().numpy(), data["target"].numpy(), equal_nan=True)
                and torch.equal(mask.cpu(), data["mask"])
            )
            result = dict(
                passed=bool(passed and unchanged),
                accepted=accepted,
                error=error,
                analytic_loss_error=loss_error,
                analytic_gradient_error=gradient_error,
                post_inputs_unchanged=bool(unchanged),
            )
    except Exception:
        result = dict(passed=False, traceback=traceback.format_exc())
    write(directory / "result.json", dict(fixture=row, **result))
    return result


def worker():
    plan = bindings()
    torch.set_num_threads(2)
    torch.xpu.set_device(1)
    if not torch.xpu.is_available():
        raise RuntimeError("XPU1 unavailable")
    write(ROOT / "WORKER_STARTED.json", dict(optimizer_updates=0))
    with np.load(FIXTURE, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    numerical = []
    for seed in SEEDS:
        for case in range(3):
            tensors = [
                torch.from_numpy(data[k])
                for k in [
                    f"x_{case}",
                    f"valid_{case}",
                    f"y_{case}",
                    f"observed_{case}",
                    "population_weights",
                    f"indices_{case}",
                ]
            ]
            numerical.append(numerical_case(f"seed_{seed}_case_{case}", seed, *tensors, plan))
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
    gen = torch.Generator().manual_seed(73)
    x = torch.randn(128, 32, 64, generator=gen)
    v = torch.ones_like(x, dtype=torch.bool)
    v[-1, -1, -1] = False
    x[-1, -1, -1] = float("nan")
    y = torch.randn(128, 3, 64, generator=gen)
    m = torch.ones_like(y, dtype=torch.bool)
    m[-1, -1, -1] = False
    y[-1, -1, -1] = float("nan")
    w = torch.full((128,), 1 / 128, dtype=torch.float64)
    indices = torch.arange(128)
    numerical.append(numerical_case("synthetic_full_batch", 101, x, v, y, m, w, indices, plan))
    torch.manual_seed(101)
    model = (
        GuardedPoolTemporalPredictionEngine(64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16))
        .to("xpu:1")
        .eval()
    )
    faults = []
    for index, row in enumerate(plan["fault_schedule"]):
        faults.append(fault_case(index, row, model, plan))
        if (index + 1) % 48 == 0:
            print(
                f"Fault-injection cases{index + 1}/288;failures={sum(not r['passed'] for r in faults)}",
                flush=True,
            )
    bindings()
    result = dict(
        completed=True,
        numerical_cases=len(numerical),
        numerical_passes=sum(r["passed"] for r in numerical),
        fault_cases=len(faults),
        fault_passes=sum(r["passed"] for r in faults),
        integration_candidate_passed=all(r["passed"] for r in numerical + faults),
        numerical_results=numerical,
        production_entrypoints_changed=False,
        scientific_optimizer_updates=0,
        scientific_training_started=False,
        internal_evaluation_scoring=False,
        corrected_replication_authorized=False,
        replication_gate="UNEVALUATED",
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    write(ROOT / "worker_summary.json", result)


def run():
    plan = bindings()
    if (ROOT / "STARTED.json").exists():
        raise FileExistsError("Existing candidate attempt preserved;no retry")
    if (
        os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384"
        or "DNNL_PRIMITIVE_CACHE_CAPACITY" in os.environ
    ):
        raise ValueError("Frozen cache policy required")
    write(
        ROOT / "STARTED.json",
        dict(plan_sha256=sha(ROOT / "plan.json"), scientific_optimizer_updates=0),
    )
    process = run_bounded(
        [sys.executable, str(Path(__file__).resolve()), "worker"],
        ROOT / "console.log",
        timeout=plan["worker_timeout_seconds"],
        cleanup=plan["kill_cleanup_seconds"],
    )
    complete = (ROOT / "worker_summary.json").is_file()
    result = dict(
        process=process,
        worker_completed=complete,
        integration_candidate_passed=complete
        and not process["timed_out"]
        and process["returncode"] == 0
        and json.loads((ROOT / "worker_summary.json").read_text())["integration_candidate_passed"],
        scientific_optimizer_updates=0,
        production_entrypoints_changed=False,
        corrected_replication_authorized=False,
    )
    write(ROOT / "summary.json", result)
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.rglob("*") if p.is_file()})
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "run", "worker"])
    args = p.parse_args()
    {"prepare": prepare, "run": run, "worker": worker}[args.action]()


if __name__ == "__main__":
    main()
