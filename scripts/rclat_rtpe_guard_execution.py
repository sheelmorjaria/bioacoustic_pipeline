"""Bounded two-device optimiser/history engineering check; no scientific run."""

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
from bioacoustic_pipeline.diagnostics.rclat_guard_execution import optimizer, state_vector
from bioacoustic_pipeline.diagnostics.rclat_mask_stages import run_bounded
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write
from bioacoustic_pipeline.representation.rclat_rtpe_pool import PoolTemporalPredictionEngine
from bioacoustic_pipeline.representation.rclat_rtpe_guarded import (
    GuardedPoolTemporalPredictionEngine,
    guarded_minibatch_mse,
)
from bioacoustic_pipeline.representation.rclat_rtpe_weighted import minibatch_mse

ROOT = Path("outputs/rclat_rtpe_guard_execution_v1")
FIXTURE = Path("outputs/rclat_rtpe1_protocol_v2/training_numerical_fixtures.npz")
SEEDS = [101, 211, 307, 401, 503]


def prepare():
    ROOT.mkdir(exist_ok=False)
    paths = [
        Path(__file__),
        FIXTURE,
        Path("src/bioacoustic_pipeline/diagnostics/rclat_guard_execution.py"),
        Path("tests/test_rclat_guard_execution.py"),
    ]
    for p in [
        "rclat_rtpe.py",
        "rclat_rtpe_pool.py",
        "rclat_rtpe_guarded.py",
        "rclat_rtpe_weighted.py",
    ]:
        paths.append(Path("src/bioacoustic_pipeline/representation") / p)
    write(
        ROOT / "plan.json",
        dict(
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            seeds=SEEDS,
            devices=["xpu:0", "xpu:1"],
            updates_per_seed=20,
            clip=5.0,
            relative_tolerance=1e-4,
            CPU_identity="bit-identical",
            optimizer="AdamW lr .001, weight_decay .0001, default betas/eps, foreach=False,fused=False",
            history="20 consecutive updates per model;real cases step modulo 3;batch128 at every fifth step;no cache emptying between steps;new forward/backward each step",
            metrics=[
                "loss",
                "forecast",
                "parameter_gradient",
                "input_gradient",
                "parameters",
                "exp_avg",
                "exp_avg_sq",
                "step",
            ],
            worker_limit_seconds=300,
            no_retry=True,
            scientific_updates=0,
            evaluation_scoring=False,
            production_adoption=False,
            dependencies={str(p): sha(p) for p in paths},
        ),
    )


def bindings():
    p = json.loads((ROOT / "plan.json").read_text())
    assert all(sha(k) == v for k, v in p["dependencies"].items())
    return p


def vector(model, gradient=False):
    return torch.cat(
        [
            (p.grad if gradient else p).detach().cpu().flatten()
            for p in model.parameters()
            if not gradient or p.grad is not None
        ]
    )


def error(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float((a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-30))


def step(model, opt, loss_fn, batch):
    x, v, y, m, w, indices = batch
    leaf = x.detach().requires_grad_()
    opt.zero_grad(set_to_none=True)
    forecast = model(leaf, v)["forecast"][:, -1]
    loss = loss_fn(forecast, y, m, w, indices)
    loss.backward()
    grads = vector(model, True)
    ig = leaf.grad.detach().cpu()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 5.0, foreach=False, error_if_nonfinite=True
    )
    opt.step()
    return dict(
        loss=loss.detach().cpu().reshape(1),
        forecast=forecast.detach().cpu(),
        parameter_gradient=grads,
        input_gradient=ig,
        parameters=vector(model),
        exp_avg=state_vector(opt, "exp_avg"),
        exp_avg_sq=state_vector(opt, "exp_avg_sq"),
        step=state_vector(opt, "step"),
        clip_norm=norm.detach().cpu().reshape(1),
    )


def sequence(device, seed, plan):
    d = ROOT / f"{device.replace(':', '_')}_seed_{seed}"
    d.mkdir(exist_ok=False)
    write(d / "STARTED.json", dict(device=device, seed=seed))
    torch.manual_seed(seed)
    original = PoolTemporalPredictionEngine(64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16))
    cpu = GuardedPoolTemporalPredictionEngine(64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16))
    cpu.load_state_dict(original.state_dict())
    gpu = copy.deepcopy(cpu).to(device)
    models = [original, cpu, gpu]
    opts = [optimizer(m.parameters()) for m in models]
    with np.load(FIXTURE) as f:
        data = {k: f[k] for k in f.files}
    gen = torch.Generator().manual_seed(73)
    x = torch.randn(128, 32, 64, generator=gen)
    v = torch.ones_like(x, dtype=torch.bool)
    v[-1, -1, -1] = False
    x[-1, -1, -1] = float("nan")
    y = torch.randn(128, 3, 64, generator=gen)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[-1, -1, -1] = False
    y[-1, -1, -1] = float("nan")
    full = [x, v, y, mask, torch.full((128,), 1 / 128, dtype=torch.float64), torch.arange(128)]
    maxima = {k: 0.0 for k in plan["metrics"] + ["clip_norm"]}
    for update in range(1, 21):
        c = (update - 1) % 3
        batch = (
            full
            if update % 5 == 0
            else [
                torch.from_numpy(data[k])
                for k in [
                    f"x_{c}",
                    f"valid_{c}",
                    f"y_{c}",
                    f"observed_{c}",
                    "population_weights",
                    f"indices_{c}",
                ]
            ]
        )
        xb = [t.to(device) if i < 4 else t for i, t in enumerate(batch)]
        a = step(original, opts[0], minibatch_mse, batch)
        b = step(cpu, opts[1], guarded_minibatch_mse, batch)
        z = step(gpu, opts[2], guarded_minibatch_mse, xb)
        assert all(torch.equal(a[k], b[k]) for k in a), "CPU guard-only update differs"
        errors = {k: error(a[k], z[k]) for k in a}
        passed = max(errors.values()) < plan["relative_tolerance"]
        write(
            d / f"update_{update:02}.json",
            dict(update=update, CPU_bit_identical=True, errors=errors, passed=passed),
        )
        for k, e in errors.items():
            maxima[k] = max(maxima[k], e)
        if not passed:
            raise ValueError(f"Update {update} exceeds frozen tolerance: {errors}")
    torch.save(dict(original_CPU=a, candidate_CPU=b, candidate_XPU=z), d / "final_tensors.pt")
    result = dict(
        passed=True, device=device, seed=seed, engineering_updates_per_model=20, max_errors=maxima
    )
    write(d / "result.json", result)
    print(device, seed, "PASS", flush=True)
    return result


def worker(device):
    plan = bindings()
    torch.set_num_threads(2)
    results = []
    for seed in SEEDS:
        try:
            results.append(sequence(device, seed, plan))
        except Exception:
            write(
                ROOT / f"{device.replace(':', '_')}_FAILURE.json",
                dict(seed=seed, traceback=traceback.format_exc()),
            )
            raise
    write(
        ROOT / f"{device.replace(':', '_')}_summary.json",
        dict(passed=True, sequences=results, scientific_updates=0),
    )


def run():
    p = bindings()
    assert (
        os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") == "16384"
        and "DNNL_PRIMITIVE_CACHE_CAPACITY" not in os.environ
    )
    write(ROOT / "STARTED.json", dict(plan_sha=sha(ROOT / "plan.json")))
    results = []
    for device in p["devices"]:
        result = run_bounded(
            [sys.executable, str(Path(__file__).resolve()), "worker", "--device", device],
            ROOT / f"{device.replace(':', '_')}.log",
            timeout=300,
            cleanup=3,
        )
        results.append(dict(device=device, **result))
        if result["returncode"] != 0 or result["timed_out"]:
            break
    passed = len(results) == 2 and all(r["returncode"] == 0 and not r["timed_out"] for r in results)
    write(
        ROOT / "summary.json",
        dict(passed=passed, processes=results, scientific_updates=0, replication_restarted=False),
    )
    write(ROOT / "manifest.json", {str(q): sha(q) for q in ROOT.rglob("*") if q.is_file()})
    print(json.dumps(dict(passed=passed, processes=results), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "worker"])
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.device)
    else:
        {"prepare": prepare, "run": run}[args.action]()
