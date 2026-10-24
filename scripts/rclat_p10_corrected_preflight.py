"""Prospective, bounded engineering validation of version4; no validation scoring."""

import argparse
import json
import math
from pathlib import Path
import time
import traceback

import torch

import rclat_p10_numerical_diagnostic as prior
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text
from bioacoustic_pipeline.representation.rclat_numerics import (
    check_gradients,
    check_parameters_and_optimizer,
    gradient_norm_float64,
)
from bioacoustic_pipeline.representation.rclat_scale_v3 import ScaleCTC as Original
from bioacoustic_pipeline.representation.rclat_scale_v4 import ScaleCTC, stable_softmax

OUT = Path("outputs/rclat_p10_corrected_preflight_v1")
CONFIG = {"width": 256, "layers": 12, "heads": 4, "expansion": 1024, "kernel": 15, "dropout": 0.0}


def prepare():
    spec, cohort, _ = prior.check()
    ordered = prior.original.epoch_order(cohort["selected_training"], 1)
    batches = [ordered[i : i + 4] for i in range(0, len(ordered), 4)]
    ranked = sorted(
        range(len(batches)), key=lambda i: (max(r["ctc_frames"] for r in batches[i]), i)
    )
    selected = sorted({0, 2, ranked[0], ranked[len(ranked) // 2], ranked[-1]})
    paths = [
        Path(__file__),
        Path("src/bioacoustic_pipeline/representation/rclat_scale_v4.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_numerics.py"),
        Path("tests/test_rclat_scale_v4.py"),
        prior.OUT / "closeout.json",
    ]
    minimum = min(r["ctc_frames"] for r in ordered)
    maximum = max(r["ctc_frames"] for r in ordered)
    shapes = [(1, minimum), (2, 128)] + [
        (4, n) for n in [256, 299, 320, 333, 334, 335, 336, 352, 384, maximum]
    ]
    prior.original.protocol.freeze(
        OUT / "plan.json",
        {
            "scope": "Version4 engineering only. Original P10 closed; no scientific validation/final-test scoring or restart.",
            "devices": {a: v["device"] for a, v in spec["frontends"].items()},
            "configuration": CONFIG,
            "seed": 0,
            "synthetic_shapes_batch_length": shapes,
            "heads": 4,
            "full_model_cases": [
                {
                    "original_epoch1_step": i + 1,
                    "uids": [r["uid"] for r in batches[i]],
                    "ctc_lengths": [r["ctc_frames"] for r in batches[i]],
                }
                for i in selected
            ],
            "full_model_reference": "Original v3 CPU FP32 with shared initial state; actual CPU CTC. Corrected v4 XPU FP32 with identical state/input/targets.",
            "tolerances": {
                "synthetic_forward_relative_L2": 1e-5,
                "synthetic_gradient_relative_L2": 1e-5,
                "full_model_logits_max_absolute": 1e-4,
                "full_model_gradient_relative_L2": 1e-3,
                "norm_relative_error": 1e-4,
            },
            "repeated_optimizer_updates_per_arm": 200,
            "step_order": "First200 original epoch1 batches, paired across arms; fresh seed0 model, original AdamW/lr schedule/clip5. No stopping based on loss decrease.",
            "checks_each_step": "Finite logits/loss, every parameter gradient before clip, finite clip norm, every parameter and optimizer tensor after update; step+UID journal.",
            "independent_FP64_norm_checks": "Steps1,3,and every25th, before clip",
            "abstention": "Stop first failed check;preserve failure;no adaptive batch,precision,seed or policy changes.",
            "interpretation": "Passing is bounded numerical readiness,not30epoch stability or speech-recognition utility.",
            "dependencies": {str(p): prior.original.protocol.sha(p) for p in paths},
        },
    )
    print("Corrected version4 engineering plan fixed; no model run.")


def check():
    spec, cohort, _ = prior.check()
    plan = json.loads((OUT / "plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if prior.original.protocol.sha(path) != digest:
            raise ValueError("Version4 engineering dependency changed")
    return spec, cohort, plan


def relative_error(observed, expected):
    return float(
        torch.linalg.vector_norm(observed.double() - expected.double())
        / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
    )


def vector_grad(model):
    return torch.cat([p.grad.detach().cpu().reshape(-1).double() for p in model.parameters()])


def run(arm):
    spec, cohort, plan = check()
    prior.original.protocol.freeze(
        OUT / f"{arm}_STARTED.json",
        {"plan_sha256": prior.original.protocol.sha(OUT / "plan.json"), "engineering_only": True},
    )
    device = plan["devices"][arm]
    prior.seed(device)
    generator = torch.Generator().manual_seed(0)
    tolerance = plan["tolerances"]
    synthetic = []
    for batch, length in plan["synthetic_shapes_batch_length"]:
        values = 0.5 * torch.randn(batch, 4, length, length, generator=generator)
        upstream = 0.01 * torch.randn(values.shape, generator=generator)
        lengths = torch.tensor([max(1, round(length * f)) for f in [1.0, 0.76, 0.65, 0.83][:batch]])
        valid = torch.arange(length)[None] < lengths[:, None]
        values = values.masked_fill(~valid[:, None, None, :], -10000.0)
        reference = values.double().softmax(-1)
        gradient = reference * (
            upstream.double() - (upstream.double() * reference).sum(-1, keepdim=True)
        )
        leaf = values.to(device).requires_grad_()
        output = stable_softmax(leaf)
        output.backward(upstream.to(device))
        forward_error = relative_error(output.detach().cpu(), reference)
        gradient_error = relative_error(leaf.grad.cpu(), gradient)
        if not torch.isfinite(output).all() or not torch.isfinite(leaf.grad).all():
            raise FloatingPointError("Nonfinite synthetic values")
        if (
            forward_error > tolerance["synthetic_forward_relative_L2"]
            or gradient_error > tolerance["synthetic_gradient_relative_L2"]
        ):
            raise ValueError("Synthetic CPU/XPU parity outside frozen tolerance")
        synthetic.append(
            {
                "batch": batch,
                "length": length,
                "forward_relative_L2": forward_error,
                "gradient_relative_L2": gradient_error,
            }
        )
    print(json.dumps({"arm": arm, "synthetic_cases_passed": len(synthetic)}), flush=True)
    dimensions = spec["frontends"][arm]["dimensions"]
    mapping = {r["uid"]: r for r in cohort["selected_training"]}
    parity = []
    for case in plan["full_model_cases"]:
        rows = [mapping[uid] for uid in case["uids"]]
        targets = [encode_text(r["transcript"]) for r in rows]
        prior.seed("cpu")
        cpu_model = Original(dimensions, **CONFIG)
        state = {k: v.detach().clone() for k, v in cpu_model.state_dict().items()}
        prior.seed("cpu")
        cpu_candidate = ScaleCTC(dimensions, **CONFIG)
        if any(not torch.equal(state[k], v) for k, v in cpu_candidate.state_dict().items()):
            raise ValueError("Version4 initialization changed")
        del cpu_candidate
        cpu_logits, lengths = cpu_model(*prior.batch_for(rows, arm, "cpu"))
        cpu_loss = ctc_objective(cpu_logits, lengths, targets)
        cpu_loss.backward()
        check_gradients(cpu_model)
        reference_grad = vector_grad(cpu_model)
        gpu_model = ScaleCTC(dimensions, **CONFIG).to(device)
        gpu_model.load_state_dict(state)
        logits, gpu_lengths = gpu_model(*prior.batch_for(rows, arm, device))
        loss = ctc_objective(logits, gpu_lengths, targets)
        loss.backward()
        check_gradients(gpu_model)
        if not torch.equal(lengths, gpu_lengths):
            raise ValueError("Temporal output lengths changed")
        logit_error = float((logits.detach().cpu() - cpu_logits.detach()).abs().max())
        grad_error = relative_error(vector_grad(gpu_model), reference_grad)
        if (
            logit_error > tolerance["full_model_logits_max_absolute"]
            or grad_error > tolerance["full_model_gradient_relative_L2"]
        ):
            raise ValueError("Full model CPU/XPU parity outside tolerance")
        parity.append(
            {
                **case,
                "logits_max_abs_error": logit_error,
                "gradient_relative_L2_error": grad_error,
                "cpu_ctc_loss": float(cpu_loss.detach()),
                "xpu_ctc_loss": float(loss.detach()),
            }
        )
        print(json.dumps({"arm": arm, "parity": parity[-1]}), flush=True)
        del cpu_model, gpu_model, reference_grad, cpu_logits, logits, cpu_loss, loss, state
        torch.xpu.empty_cache()
    prior.seed(device)
    model = ScaleCTC(dimensions, **CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    ordered = prior.original.epoch_order(cohort["selected_training"], 1)
    total_steps = 30 * math.ceil(len(ordered) / 4)
    trace = []
    torch.xpu.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with (OUT / f"{arm}_steps.jsonl").open("x", buffering=1) as stream:
        for step in range(1, plan["repeated_optimizer_updates_per_arm"] + 1):
            rows = ordered[(step - 1) * 4 : step * 4]
            before = {"step": step, "uids": [r["uid"] for r in rows], "event": "begin"}
            stream.write(json.dumps(before) + "\n")
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = prior.original.learning_rate(step, total_steps)
            logits, lengths = model(*prior.batch_for(rows, arm, device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite model logits")
            loss = ctc_objective(logits, lengths, [encode_text(r["transcript"]) for r in rows])
            loss.backward()
            checked = check_gradients(model)
            reference_norm = (
                gradient_norm_float64(model) if step in [1, 3] or step % 25 == 0 else None
            )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            norm_error = (
                abs(float(norm) - reference_norm) / max(reference_norm, 1e-30)
                if reference_norm is not None
                else None
            )
            if norm_error is not None and norm_error > tolerance["norm_relative_error"]:
                raise ValueError("Gradient norm differs from independent FP64 reference")
            optimizer.step()
            state_checks = check_parameters_and_optimizer(model, optimizer)
            row = {
                "step": step,
                "uids": before["uids"],
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "gradient_tensors_checked": checked,
                "reference_norm_float64": reference_norm,
                "norm_relative_error": norm_error,
                **state_checks,
            }
            trace.append(row)
            stream.write(json.dumps({**row, "event": "completed"}) + "\n")
            if step % 25 == 0:
                print(
                    json.dumps(
                        {"arm": arm, "completed_engineering_steps": step, "loss": row["loss"]}
                    ),
                    flush=True,
                )
    torch.xpu.synchronize(device)
    prior.original.protocol.freeze(
        OUT / f"{arm}_results.json",
        {
            "arm": arm,
            "device": device,
            "plan_sha256": prior.original.protocol.sha(OUT / "plan.json"),
            "engineering_gate_passed": True,
            "synthetic_parity": synthetic,
            "full_model_parity": parity,
            "repeated_optimizer_updates": len(trace),
            "step_trace": trace,
            "wall_seconds_200_steps_including_io": time.perf_counter() - started,
            "peak_torch_allocated_bytes": torch.xpu.max_memory_allocated(device),
            "all_gradient_parameter_optimizer_state_checks_passed": True,
            "scientific_validation_or_test_scored": False,
            "closed_P10_resumed": False,
            "limitation": "Bounded200step engineering validation;not30epoch stability or transcription utility.",
        },
    )
    print(f"{arm}: version4 engineering checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--arm", choices=["mel", "combined"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.arm is None:
        parser.error("arm required")
    else:
        try:
            run(args.arm)
        except Exception:
            path = OUT / f"{args.arm}_FAILURE.json"
            if not path.exists():
                prior.original.protocol.freeze(
                    path, {"traceback": traceback.format_exc(), "automatic_retry_permitted": False}
                )
            raise
