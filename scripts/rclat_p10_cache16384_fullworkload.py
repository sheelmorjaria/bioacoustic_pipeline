"""Frozen real-workload cache engineering validation; no transcript scoring/C3."""

import argparse
import copy
import ctypes
import json
import math
from pathlib import Path
import platform
import random
import time
import traceback
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import rclat_p10_c2_train as predecessor
from rclat_p10_train import protocol, sequence, epoch_order, learning_rate
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.representation.rclat_scale_v4 import ScaleCTC
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text
from bioacoustic_pipeline.representation.rclat_numerics import (
    check_gradients,
    check_parameters_and_optimizer,
)
from rclat_p10_c1_shape_replay import telemetry
from rclat_p10_memory_attribution import Mallinfo2

OUT = Path("outputs/rclat_p10_cache16384_fullworkload_engineering_v1")
CHECKPOINTS = Path("checkpoints/rclat_p10_cache16384_fullworkload_engineering_v1")


def check():
    predecessor.check()
    plan = json.loads((OUT / "plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if protocol.sha(path) != digest:
            raise ValueError(f"Engineering dependency changed: {path}")
    if os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384":
        raise ValueError("Required cache policy absent")
    if os.environ.get("DNNL_PRIMITIVE_CACHE_CAPACITY") is not None:
        raise ValueError("Legacy cache override")
    if os.environ.get("ONEDNN_VERBOSE") != "profile_create,filter=conv":
        raise ValueError("Required logging absent")
    return plan, json.loads((predecessor.original.OUT / "cohort.json").read_text())


def build(dim):
    return ScaleCTC(dim, width=256, layers=12, heads=4, expansion=1024, kernel=15, dropout=0.0)


def execute(action, arm):
    plan, cohort = check()
    marker = OUT / f"{arm}_{action}_STARTED.json"
    protocol.freeze(
        marker,
        dict(
            action=action,
            arm=arm,
            plan_sha256=protocol.sha(OUT / "plan.json"),
            scientific_execution=False,
            stage2_ready=False,
        ),
    )
    device = plan["frontends"][arm]["device"]
    dim = plan["frontends"][arm]["dimensions"]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    torch.xpu.set_device(device)
    scaler = json.loads((predecessor.original.OUT / f"{arm}_scaler.json").read_text())

    def batch(rows, where):
        arrays = [frozen_scale(*sequence(r["uid"], arm), scaler) for r in rows]
        return (
            pad_sequence([torch.from_numpy(x) for x, m in arrays], batch_first=True).to(where),
            pad_sequence([torch.from_numpy(m) for x, m in arrays], batch_first=True).to(where),
            torch.tensor([len(x) for x, m in arrays]),
        )

    training = cohort["selected_training"]
    validation = sorted(cohort["selected_validation"], key=lambda r: r["uid"])
    ordered = epoch_order(training, 1)
    if action == "parity":
        cases = []
        for offset in [0, 4, 8]:
            rows = ordered[offset : offset + 4]
            cpu = build(dim)
            gpu = copy.deepcopy(cpu).to(device)
            inputs = batch(rows, "cpu")
            targets = [encode_text(r["transcript"]) for r in rows]
            yc, lc = cpu(*inputs)
            lossc = ctc_objective(yc, lc, targets)
            lossc.backward()
            yg, lg = gpu(inputs[0].to(device), inputs[1].to(device), inputs[2])
            lossg = ctc_objective(yg, lg, targets)
            lossg.backward()
            check_gradients(gpu)
            num = den = 0.0
            for a, b in zip(cpu.parameters(), gpu.parameters()):
                x = a.grad.double()
                y = b.grad.detach().cpu().double()
                num += float((x - y).square().sum())
                den += float(x.square().sum())
            grad = (num / max(den, 1e-300)) ** 0.5
            logits = float(
                (yc.detach().double() - yg.detach().cpu().double()).norm()
                / yc.detach().double().norm()
            )
            loss = abs(float(lossc.detach()) - float(lossg.detach())) / max(
                abs(float(lossc.detach())), 1e-30
            )
            cases.append(
                dict(
                    uids=[r["uid"] for r in rows],
                    gradient_relative_L2=grad,
                    logit_relative_L2=logits,
                    loss_relative=loss,
                    passed=max(grad, logits, loss) <= plan["parity_tolerance"],
                )
            )
            del cpu, gpu, inputs, yc, yg, lossc, lossg
        protocol.freeze(
            OUT / f"{arm}_parity.json",
            dict(
                cases=cases,
                passed=all(c["passed"] for c in cases),
                training_sources_only=True,
                scientific_scores=False,
            ),
        )
        return
    if not json.loads((OUT / f"{arm}_parity.json").read_text())["passed"]:
        raise RuntimeError("Parity gate failed")
    # Full shared-cache checksum verification before model updates.
    predecessor.verify_cache()
    model = build(dim)
    digest = predecessor.initial_digest(model)
    original = json.loads((predecessor.OUT / f"{arm}_initial_state.json").read_text())
    if digest != original["full_state_sha256"]:
        raise ValueError("Fresh initialization differs")
    protocol.freeze(
        OUT / f"{arm}_initial_state.json",
        dict(
            full_state_sha256=digest,
            common_encoder_head_sha256=predecessor.initial_digest(model, True),
            checkpoint_loaded=False,
            seed=0,
        ),
    )
    protocol.freeze(
        OUT / f"{arm}_runtime.json",
        dict(
            python=platform.python_version(),
            torch=torch.__version__,
            device=device,
            properties=str(torch.xpu.get_device_properties(device)),
            torch_config=torch.__config__.show(),
            capacity_requested=16384,
            capacity_API_readback=None,
        ),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    check_parameters_and_optimizer(model, optimizer)
    targets = {r["uid"]: encode_text(r["transcript"]) for r in training}
    total_schedule_steps = 30 * math.ceil(len(training) / 4)
    libc = ctypes.CDLL(None)
    libc.mallinfo2.restype = Mallinfo2
    start = time.monotonic()
    step = 0
    epochs = []

    def resources():
        m = libc.mallinfo2()
        return dict(
            glibc_in_use=m.uordblks, glibc_free=m.fordblks, glibc_mmap=m.hblkhd, **telemetry(device)
        )

    with (OUT / f"{arm}_steps.jsonl").open("x", buffering=1) as journal:

        def log(event, **kw):
            journal.write(json.dumps(dict(event=event, **kw)) + "\n")

        log("initial_resources", **resources())
        for epoch in range(1, 13):
            if time.monotonic() - start > plan["max_seconds"]:
                raise TimeoutError("Engineering time bound")
            print(f"ENGINEERING_TRAIN_BEGIN:{epoch}", flush=True)
            epoch_start = time.monotonic()
            model.train()
            rows_order = epoch_order(training, epoch)
            for pos in range(0, len(rows_order), 4):
                if time.monotonic() - start > plan["max_seconds"]:
                    raise TimeoutError("Engineering time bound")
                rows = rows_order[pos : pos + 4]
                step += 1
                context = dict(
                    epoch=epoch, step=step, epoch_step=pos // 4 + 1, uids=[r["uid"] for r in rows]
                )
                if step == 7751 and context["uids"] != plan["historical_failure_uids"]:
                    raise ValueError("Historical batch alignment changed")
                log("begin", **context)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(step, total_schedule_steps)
                optimizer.zero_grad(set_to_none=True)
                inputs = batch(rows, device)
                log(
                    "forward_begin",
                    **context,
                    feature_shape=list(inputs[0].shape),
                    native_lengths=inputs[2].tolist(),
                )
                if step == 1 or step % 100 == 0 or 7700 <= step <= 7800:
                    log("resources", **context, **resources())
                logits, lengths = model(*inputs)
                del inputs
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite training logits")
                loss = ctc_objective(logits, lengths, [targets[r["uid"]] for r in rows])
                loss.backward()
                gradients = check_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite gradient norm")
                log(
                    "backward_complete",
                    **context,
                    loss=float(loss.detach()),
                    gradient_norm=float(norm),
                    gradient_tensors_checked=gradients,
                )
                log("optimizer_begin", **context)
                optimizer.step()
                checks = check_parameters_and_optimizer(model, optimizer)
                log("step_complete", **context, **checks)
            print(f"ENGINEERING_VALIDATION_FORWARD_BEGIN:{epoch}", flush=True)
            model.eval()
            log("validation_begin", epoch=epoch, step=step, **resources())
            with torch.no_grad():
                for pos in range(0, len(validation), 4):
                    if time.monotonic() - start > plan["max_seconds"]:
                        raise TimeoutError("Engineering time bound")
                    rows = validation[pos : pos + 4]
                    inputs = batch(rows, device)
                    log(
                        "validation_forward_begin",
                        epoch=epoch,
                        batch=pos // 4 + 1,
                        uids=[r["uid"] for r in rows],
                        feature_shape=list(inputs[0].shape),
                        native_lengths=inputs[2].tolist(),
                    )
                    logits, lengths = model(*inputs)
                    del inputs
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Nonfinite validation-shaped forward logits")
            log("validation_end", epoch=epoch, step=step, **resources())
            folder = CHECKPOINTS / arm
            folder.mkdir(parents=True, exist_ok=True)
            rng = np.random.get_state()
            checkpoint = folder / f"engineering_epoch_{epoch:02d}.pt"
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    epoch=epoch,
                    step=step,
                    torch_rng=torch.get_rng_state(),
                    xpu_rng=torch.xpu.get_rng_state_all(),
                    python_rng=random.getstate(),
                    numpy_rng=[rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]],
                    engineering_only=True,
                    scientific_reuse_prohibited=True,
                    plan_sha256=protocol.sha(OUT / "plan.json"),
                ),
                checkpoint,
            )
            stats = dict(
                epoch=epoch,
                step=step,
                wall_seconds=time.monotonic() - epoch_start,
                checkpoint=str(checkpoint),
                checkpoint_sha256=protocol.sha(checkpoint),
                **resources(),
            )
            epochs.append(stats)
            protocol.freeze(OUT / f"{arm}_epoch_{epoch:02d}.json", stats)
            log("checkpoint_saved", **stats)
            print("ENGINEERING_EPOCH_END " + json.dumps(stats), flush=True)
    late = [r for r in epochs if r["epoch"] >= 3]
    x = np.array([r["epoch"] for r in late])
    rss = np.array([int(r["VmRSS"].split()[0]) / 1024 for r in late])
    native = np.array([r["glibc_in_use"] / 1024**2 for r in late])
    rss_slope = float(np.polyfit(x, rss, 1)[0])
    native_slope = float(np.polyfit(x, native, 1)[0])
    protocol.freeze(
        OUT / f"{arm}_result.json",
        dict(
            completed_epochs=len(epochs),
            completed_updates=step,
            historical_step_crossed=step > 7751,
            RSS_slope_MiB_per_epoch=rss_slope,
            native_slope_MiB_per_epoch=native_slope,
            engineering_gate_passed=len(epochs) == 12
            and rss_slope < plan["RSS_slope_max_MiB_per_epoch"]
            and native_slope < plan["native_slope_max_MiB_per_epoch"],
            scientific_scores=False,
            scientific_training_gate_passed=False,
            stage2_ready=False,
            epochs=epochs,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["parity", "run"])
    p.add_argument("--arm", choices=["mel", "combined"], required=True)
    a = p.parse_args()
    try:
        execute(a.action, a.arm)
    except Exception:
        failure = OUT / f"{a.arm}_{a.action}_FAILURE.json"
        if not failure.exists():
            protocol.freeze(
                failure,
                dict(
                    traceback=traceback.format_exc(),
                    automatic_retry=False,
                    scientific_execution=False,
                ),
            )
        raise
