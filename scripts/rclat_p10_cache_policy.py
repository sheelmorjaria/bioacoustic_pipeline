"""Frozen engineering cache-policy comparison; no scientific execution."""

import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from bioacoustic_pipeline.representation.rclat_scale_v4 import ScaleCTC
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective
from bioacoustic_pipeline.representation.rclat_numerics import (
    check_gradients,
    check_parameters_and_optimizer,
)
from rclat_p10_c1_shape_replay import telemetry
from rclat_p10_memory_attribution import Mallinfo2

ROOT = Path("outputs/rclat_p10_cache_policy")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def model():
    return ScaleCTC(64, width=256, layers=12, heads=4, expansion=1024, kernel=15, dropout=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["parity", "run"])
    ap.add_argument("--policy", choices=["default", "capacity8192"], required=True)
    ap.add_argument("--device", type=int, required=True)
    args = ap.parse_args()
    plan = json.loads((ROOT / "plan.json").read_text())
    for p, h in plan["dependencies"].items():
        if sha(p) != h:
            raise ValueError(f"Frozen diagnostic dependency changed: {p}")
    expected = None if args.policy == "default" else "8192"
    if os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != expected:
        raise ValueError("Cache environment mismatch")
    if os.environ.get("DNNL_PRIMITIVE_CACHE_CAPACITY") is not None:
        raise ValueError("Legacy cache override present")
    if os.environ.get("ONEDNN_VERBOSE") != "profile_create,filter=conv":
        raise ValueError("Logging mismatch")
    name = f"{args.policy}_xpu{args.device}"
    path = ROOT / f"{name}_{args.action}.json"
    marker = ROOT / f"{name}_{args.action}_STARTED.json"
    with marker.open("x") as f:
        json.dump(
            dict(
                plan_sha256=sha(ROOT / "plan.json"), requested_capacity=expected, device=args.device
            ),
            f,
        )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = f"xpu:{args.device}"
    torch.xpu.set_device(device)
    targets = [torch.tensor([2, 3, 4, 5] * 10) for _ in range(4)]
    selected = plan["lengths"]
    start = time.monotonic()
    if args.action == "parity":
        cases = []
        try:
            for lens in plan["parity_lengths"]:
                cpu = model()
                gpu = copy.deepcopy(cpu).to(device)
                features = torch.randn(4, max(lens), 64)
                valid = torch.ones_like(features, dtype=torch.bool)
                lengths = torch.tensor(lens)
                yc, lc = cpu(features, valid, lengths)
                xc = ctc_objective(yc, lc, targets)
                xc.backward()
                yg, lg = gpu(features.to(device), valid.to(device), lengths)
                xg = ctc_objective(yg, lg, targets)
                xg.backward()
                check_gradients(gpu)
                numerator = denominator = 0.0
                for a, b in zip(cpu.parameters(), gpu.parameters()):
                    ga = a.grad.double()
                    gb = b.grad.detach().cpu().double()
                    numerator += float((ga - gb).square().sum())
                    denominator += float(ga.square().sum())
                relative = (numerator / max(denominator, 1e-300)) ** 0.5
                logit_relative = float(
                    (yg.detach().cpu().double() - yc.detach().double()).norm()
                    / yc.detach().double().norm()
                )
                loss_relative = abs(float(xg.detach()) - float(xc.detach())) / max(
                    abs(float(xc.detach())), 1e-30
                )
                cases.append(
                    dict(
                        lengths=lens,
                        gradient_relative_L2=relative,
                        logit_relative_L2=logit_relative,
                        loss_relative=loss_relative,
                        passed=max(relative, logit_relative, loss_relative)
                        <= plan["parity_tolerance"],
                    )
                )
                del cpu, gpu, features, valid, yc, yg, xc, xg
            result = dict(
                cases=cases,
                passed=all(r["passed"] for r in cases),
                optimizer_updates=0,
                seconds=time.monotonic() - start,
            )
        except Exception as exc:
            result = dict(passed=False, cases=cases, error=repr(exc))
        path.write_text(json.dumps(result, indent=2) + "\n")
        return
    prerequisite = ROOT / f"{args.policy}_xpu{args.device}_parity.json"
    if not json.loads(prerequisite.read_text())["passed"]:
        raise RuntimeError("Parity gate failed")
    net = model().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=0.01)
    libc = ctypes.CDLL(None)
    libc.mallinfo2.restype = Mallinfo2
    result = dict(
        completed_updates=0, snapshots=[], policy=args.policy, requested_capacity=expected
    )
    with (ROOT / f"{name}_progress.jsonl").open("x", buffering=1) as progress:

        def snap(cycle, index):
            torch.xpu.synchronize(device)
            m = libc.mallinfo2()
            r = dict(
                cycle=cycle,
                batch=index,
                step=result["completed_updates"],
                seconds=time.monotonic() - start,
                glibc_in_use=m.uordblks,
                glibc_free=m.fordblks,
                glibc_mmap=m.hblkhd,
                **telemetry(device),
            )
            result["snapshots"].append(r)
            progress.write(json.dumps(r) + "\n")
            print("CACHE_SNAPSHOT " + json.dumps(r), flush=True)

        try:
            snap(0, 0)
            for cycle in range(1, 4):
                print(f"CACHE_CYCLE_BEGIN:{cycle}", flush=True)
                for i, lens in enumerate(selected, 1):
                    if time.monotonic() - start >= 600:
                        raise TimeoutError("Diagnostic time bound")
                    net.train()
                    optimizer.zero_grad(set_to_none=True)
                    x = torch.randn(4, max(lens), 64).to(device)
                    mask = torch.ones_like(x, dtype=torch.bool)
                    y, lengths = net(x, mask, torch.tensor(lens))
                    loss = ctc_objective(y, lengths, targets)
                    loss.backward()
                    check_gradients(net)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()
                    check_parameters_and_optimizer(net, optimizer)
                    del x, mask, y, lengths, loss
                    result["completed_updates"] += 1
                    if i % 32 == 0:
                        snap(cycle, i)
                print(f"CACHE_CYCLE_END:{cycle}", flush=True)
            result["stop_reason"] = "completed384updates"
            vector = np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])
            np.save(ROOT / f"{name}_engineering_final_vector.npy", vector)
            result["final_vector_sha256"] = sha(ROOT / f"{name}_engineering_final_vector.npy")
        except Exception as exc:
            result.update(stop_reason="exception", error=repr(exc))
    result["seconds"] = time.monotonic() - start
    result["primitive_failure"] = "could not create a primitive" in result.get("error", "")
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
