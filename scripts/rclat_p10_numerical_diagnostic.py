"""Separate training-only P10 failure localisation; never resumes the closed run.

Bounded replay uses fresh private model state, no validation, and stops at its
first numerical guard failure or at the end of epoch one. No gate is reassessed.
"""

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import rclat_p10_train as original
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text
from bioacoustic_pipeline.representation.rclat_scale_v3 import ScaleCTC

OUT = Path("outputs/rclat_p10_numerical")


def scalar(value):
    result = float(value)
    return result if math.isfinite(result) else str(result)


def tensor_stats(tensor):
    values = tensor.detach().cpu().double()
    finite = torch.isfinite(values)
    good = values[finite]
    return {
        "shape": list(values.shape),
        "nonfinite": int((~finite).sum()),
        "finite_max_abs": float(good.abs().max()) if good.numel() else None,
        "l2_float64": scalar(torch.linalg.vector_norm(values)),
    }


def gradients(model):
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            rows.append({"parameter": name, "missing": True})
        else:
            rows.append({"parameter": name, **tensor_stats(parameter.grad)})
    total = sum(r["l2_float64"] ** 2 for r in rows if isinstance(r.get("l2_float64"), float))
    nonfinite = [r["parameter"] for r in rows if r.get("nonfinite", 0)]
    return {
        "parameters": rows,
        "nonfinite_parameters": nonfinite,
        "missing_parameters": [r["parameter"] for r in rows if r.get("missing")],
        "global_l2_float64": None if nonfinite else math.sqrt(total),
    }


def prepare():
    spec, cohort = original.check()
    if (OUT / "plan.json").exists():
        raise RuntimeError("Diagnostic plan already fixed")
    rows = original.epoch_order(cohort["selected_training"], 1)
    dependencies = {
        str(p): original.protocol.sha(p)
        for p in [
            Path(__file__),
            original.OUT / "preregistration.json",
            original.OUT / "cache_integrity.json",
            original.OUT / "execution_closeout.json",
            original.OUT / "mel_scaler.json",
            original.OUT / "combined_scaler.json",
        ]
    }
    original.protocol.freeze(
        OUT / "plan.json",
        {
            "scope": "New numerical diagnosis only; closed P10 attempt preserved. No validation or final-test model input.",
            "initial_policy": "Fresh seed0 and original architecture/scalers/order/optimizer. Inspect per-parameter gradients before original clipping; compare CPU float64 norms. Stop first nonfinite guard, maximum one original training epoch. Updates, if needed to reproduce a later failure, are private diagnostic replay only.",
            "maximum_diagnostic_steps_per_arm": math.ceil(len(rows) / 4),
            "first_batch_uids": [r["uid"] for r in rows[:4]],
            "probe_policy": "On isolated failing state, no optimizer updates: compare CPU/XPU and CPU CTC FP32/FP64 logit gradients. Any additional numerical variant must be identified separately.",
            "devices": {a: v["device"] for a, v in spec["frontends"].items()},
            "dependencies": dependencies,
            "scientific_training_resumed": False,
            "stage2_ready": False,
        },
    )
    print("Separate numerical-diagnostic plan fixed; no model executed.")


def check():
    spec, cohort = original.check()
    plan = json.loads((OUT / "plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if original.protocol.sha(path) != digest:
            raise ValueError("Diagnostic dependency changed")
    return spec, cohort, plan


def seed(device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if device.startswith("xpu"):
        torch.xpu.set_device(device)
        torch.xpu.manual_seed_all(0)


def model_for(arm, device):
    return ScaleCTC(
        64 if arm == "mel" else 76,
        width=256,
        layers=12,
        heads=4,
        expansion=1024,
        kernel=15,
        dropout=0.0,
    ).to(device)


def batch_for(rows, arm, device):
    scaler = json.loads((original.OUT / f"{arm}_scaler.json").read_text())
    arrays = []
    receipt = json.loads((original.OUT / "cache_integrity.json").read_text())
    for row in rows:
        path = original.CACHE / f"{row['uid']}.npz"
        if original.protocol.sha(path) != receipt["files"][str(path)]:
            raise ValueError("Training cache changed")
        arrays.append(frozen_scale(*original.sequence(row["uid"], arm), scaler))
    lengths = torch.tensor([len(x) for x, _ in arrays])
    x = pad_sequence([torch.from_numpy(x) for x, _ in arrays], batch_first=True).to(device)
    m = pad_sequence([torch.from_numpy(m) for _, m in arrays], batch_first=True).to(device)
    return x, m, lengths


def locate(arm):
    spec, cohort, plan = check()
    marker = OUT / f"{arm}_LOCATE_STARTED.json"
    original.protocol.freeze(
        marker, {"diagnostic_plan_sha256": original.protocol.sha(OUT / "plan.json")}
    )
    device = spec["frontends"][arm]["device"]
    seed(device)
    model = model_for(arm, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    ordered = original.epoch_order(cohort["selected_training"], 1)
    steps = 30 * math.ceil(len(ordered) / 4)
    trace = []
    for pos in range(0, len(ordered), 4):
        step = pos // 4 + 1
        rows = ordered[pos : pos + 4]
        for group in optimizer.param_groups:
            group["lr"] = original.learning_rate(step, steps)
        optimizer.zero_grad(set_to_none=True)
        inputs = batch_for(rows, arm, device)
        logits, lengths = model(*inputs)
        targets = [encode_text(r["transcript"]) for r in rows]
        loss = ctc_objective(logits, lengths, targets)
        loss.backward()
        stats = gradients(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        entry = {
            "step": step,
            "uids": [r["uid"] for r in rows],
            "input_lengths": inputs[2].tolist(),
            "ctc_lengths": lengths.tolist(),
            "targets": [len(t) for t in targets],
            "logits": tensor_stats(logits),
            "loss": scalar(loss.detach()),
            "gradient_norm_guard": scalar(norm.detach()),
            "global_l2_float64": stats["global_l2_float64"],
            "nonfinite_parameters": stats["nonfinite_parameters"],
        }
        trace.append(entry)
        print(json.dumps(entry), flush=True)
        if not torch.isfinite(norm):
            torch.save(
                {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "step": step,
                    "uids": entry["uids"],
                    "diagnostic_only": True,
                },
                OUT / f"{arm}_isolated_state.pt",
            )
            original.protocol.freeze(
                OUT / f"{arm}_location.json",
                {
                    "diagnostic_plan_sha256": original.protocol.sha(OUT / "plan.json"),
                    "device": device,
                    "failure_reproduced": True,
                    "failing_step": step,
                    "diagnostic_optimizer_updates": step - 1,
                    "trace": trace,
                    "pre_clip_gradients": stats,
                    "validation_used": False,
                    "closed_p10_resumed": False,
                },
            )
            return
        optimizer.step()
    original.protocol.freeze(
        OUT / f"{arm}_location.json",
        {
            "failure_reproduced": False,
            "diagnostic_optimizer_updates": len(trace),
            "trace": trace,
            "validation_used": False,
            "closed_p10_resumed": False,
        },
    )


def probe(arm, device):
    _, cohort, _ = check()
    saved = torch.load(OUT / f"{arm}_isolated_state.pt", map_location="cpu", weights_only=True)
    mapping = {r["uid"]: r for r in cohort["selected_training"]}
    rows = [mapping[uid] for uid in saved["uids"]]
    tag = device.replace(":", "_")
    original.protocol.freeze(
        OUT / f"{arm}_{tag}_PROBE_STARTED.json", {"no_optimizer_updates": True}
    )
    seed(device)
    model = model_for(arm, device)
    model.load_state_dict(saved["model"])
    model.train()
    inputs = batch_for(rows, arm, device)
    logits, lengths = model(*inputs)
    targets = [encode_text(r["transcript"]) for r in rows]
    comparisons = []
    for dtype in [torch.float32, torch.float64]:
        copied = logits.detach().cpu().to(dtype).requires_grad_()
        loss = ctc_objective(copied, lengths, targets)
        (gradient,) = torch.autograd.grad(loss, copied)
        comparisons.append(
            {
                "dtype": str(dtype),
                "loss": scalar(loss.detach()),
                "logit_gradient": tensor_stats(gradient),
            }
        )
    logits.retain_grad()
    loss = ctc_objective(logits, lengths, targets)
    loss.backward()
    report = {
        "arm": arm,
        "device": device,
        "failing_step": saved["step"],
        "uids": saved["uids"],
        "logits": tensor_stats(logits),
        "ctc_precision_comparisons": comparisons,
        "backward_logit_gradient": tensor_stats(logits.grad),
        "parameter_gradients": gradients(model),
        "optimizer_updates": 0,
        "validation_used": False,
        "closed_p10_resumed": False,
    }
    torch.save(
        {
            "logits": logits.detach().cpu(),
            "logit_gradient": logits.grad.detach().cpu(),
            "uids": saved["uids"],
            "diagnostic_only": True,
        },
        OUT / f"{arm}_{tag}_probe_tensors.pt",
    )
    original.protocol.freeze(OUT / f"{arm}_{tag}_probe.json", report)
    print(
        json.dumps(
            {
                "device": device,
                "ctc": comparisons,
                "nonfinite_parameters": report["parameter_gradients"]["nonfinite_parameters"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["prepare", "locate", "probe"])
    parser.add_argument("--arm", choices=["mel", "combined"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.arm is None:
        parser.error("arm required")
    elif args.action == "locate":
        locate(args.arm)
    else:
        probe(args.arm, args.device)
