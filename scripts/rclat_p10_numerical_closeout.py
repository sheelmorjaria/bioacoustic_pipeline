"""Assemble verified local numerical evidence; no models or optimizers run."""

import json
from pathlib import Path
import time

import torch

import rclat_p10_numerical_diagnostic as diagnosis


def analytic_reference_checks():
    """Validate CPU-autograd baseline against independent analytic FP64 VJP."""
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(0)
    rows = []
    for length in [299, 320, 333, 334, 335, 336, 352, 384]:
        scores = 0.5 * torch.randn(4, 4, length, length, generator=generator)
        upstream = 0.01 * torch.randn(scores.shape, generator=generator)
        lengths = torch.tensor(
            [length, round(length * 0.76), round(length * 0.65), round(length * 0.83)]
        )
        valid = torch.arange(length)[None] < lengths[:, None]
        scores = scores.masked_fill(~valid[:, None, None, :], -10000)
        leaf = scores.clone().requires_grad_()
        leaf.softmax(-1).backward(upstream)
        probabilities = scores.double().softmax(-1)
        analytic = probabilities * (
            upstream.double() - (upstream.double() * probabilities).sum(-1, keepdim=True)
        )
        error = float(
            torch.linalg.vector_norm(leaf.grad.double() - analytic)
            / torch.linalg.vector_norm(analytic)
        )
        if error >= 1e-5:
            raise ValueError("CPU reference failed analytic VJP cross-check")
        rows.append({"length": length, "cpu_FP32_relative_L2_error_vs_analytic_FP64": error})
    return rows


def main():
    _, _, plan = diagnosis.check()
    out = diagnosis.OUT
    arms = {}
    for arm, index in [("mel", 0), ("combined", 1)]:
        location = json.loads((out / f"{arm}_location.json").read_text())
        cpu = json.loads((out / f"{arm}_cpu_probe.json").read_text())
        xpu = json.loads((out / f"{arm}_xpu_{index}_probe.json").read_text())
        attention = json.loads((out / f"{arm}_attention_operator_results.json").read_text())
        full = json.loads((out / f"{arm}_composite_full_model.json").read_text())
        if not location["failure_reproduced"] or location["failing_step"] != 3:
            raise ValueError("Replay summary differs")
        if (
            cpu["parameter_gradients"]["nonfinite_parameters"]
            or not xpu["parameter_gradients"]["nonfinite_parameters"]
        ):
            raise ValueError("CPU/XPU localisation summary differs")
        for record in [cpu, xpu]:
            if record["backward_logit_gradient"]["nonfinite"] or any(
                r["logit_gradient"]["nonfinite"] for r in record["ctc_precision_comparisons"]
            ):
                raise ValueError("CTC isolation summary differs")
        native = next(
            r
            for r in attention["reports"]
            if r["device"].startswith("xpu") and r["variant"] == "native"
        )
        composite = next(r for r in attention["reports"] if r["variant"] == "composite_softmax")
        candidates = [r for r in full["reports"] if r["variant"] == "composite"]
        if any(r["nonfinite_parameters"] for r in candidates):
            raise ValueError("Candidate still has nonfinite gradients")
        arms[arm] = {
            "reproduced_failing_step": location["failing_step"],
            "private_diagnostic_optimizer_updates": location["diagnostic_optimizer_updates"],
            "failing_training_uids": location["trace"][-1]["uids"],
            "nonfinite_parameters_in_location": len(
                location["pre_clip_gradients"]["nonfinite_parameters"]
            ),
            "nonfinite_parameters_in_cpu_probe": 0,
            "nonfinite_parameters_in_xpu_probe": len(
                xpu["parameter_gradients"]["nonfinite_parameters"]
            ),
            "ctc_FP32_and_FP64_logit_gradients_finite": True,
            "native_attention_softmax_VJP_relative_L2_error_vs_CPU": native["comparisons"][
                "masked_scores"
            ]["gradient"]["relative_l2_difference_from_cpu"],
            "composite_attention_softmax_VJP_relative_L2_error_vs_CPU": composite["comparisons"][
                "masked_scores"
            ]["gradient"]["relative_l2_difference_from_cpu"],
            "full_model_candidate": [
                {
                    k: r[k]
                    for k in [
                        "device",
                        "backward",
                        "loss",
                        "logits_max_abs_error_vs_cpu",
                        "global_gradient_l2_float64",
                        "global_gradient_relative_l2_error_vs_cpu",
                        "nonfinite_parameters",
                    ]
                }
                for r in candidates
            ],
        }
    random_cases = []
    for index in [0, 1]:
        record = json.loads((out / f"random_softmax_xpu_{index}.json").read_text())
        random_cases.extend(r for r in record["reports"] if r["device"] != "cpu")
    native_errors = [
        r["relative_l2_gradient_error_vs_cpu"] for r in random_cases if r["mode"] == "native"
    ]
    composite_errors = [
        r["relative_l2_gradient_error_vs_cpu"] for r in random_cases if r["mode"] == "composite"
    ]
    previous = json.loads((diagnosis.original.OUT / "execution_closeout.json").read_text())
    if previous["joint_learning_gate_passed"] or previous["final_test_scored"]:
        raise ValueError("Closed scientific outcome changed")
    if any(diagnosis.original.OUT.glob("*_epoch_*.json")):
        raise ValueError("Unexpected scientific continuation")
    analytic = analytic_reference_checks()
    files = sorted(out.glob("*.json")) + sorted(out.glob("*.pt")) + sorted(out.glob("*.log"))
    scripts = [
        Path(__file__),
        Path("scripts/rclat_p10_numerical_diagnostic.py"),
        Path("scripts/rclat_p10_attention_diagnostic.py"),
        Path("scripts/rclat_p10_softmax_diagnostic.py"),
        Path("scripts/rclat_xpu_softmax_reproducer.py"),
        Path("tests/test_rclat_numerical_diagnostics.py"),
    ]
    result = {
        "status": "COMPLETED_TRAINING_ONLY_NUMERICAL_DIAGNOSTIC",
        "closed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conclusion": "Observed P10 failure localised to incorrect native XPU FP32 softmax backward in this installed environment. Dataset-free synthetic cases reproduce on both B60s. Exact lower-level kernel/driver defect not established.",
        "norm_overflow_only_explanation_excluded": True,
        "arms": arms,
        "dataset_free_reproducer": {
            "native_XPU_cases": len(native_errors),
            "tested_last_dimensions_per_XPU": [299, 320, 333, 334, 335, 336, 352, 384],
            "native_relative_L2_error_range": [min(native_errors), max(native_errors)],
            "composite_relative_L2_error_max": max(composite_errors),
            "CPU_reference_analytic_checks": analytic,
        },
        "candidate": "Stable composite max/exp/sum softmax, tested only in isolated diagnostic copies; no frozen production module changed.",
        "limits": [
            "Two isolated full-model training cases do not establish30epoch stability or transcription performance.",
            "Private replays locate step3; original run lacked per-step logging, so historical step identification relies on reproduction.",
            "Finite-gradient checks and four-example memorisation did not guarantee gradient correctness; backend parity checks are needed.",
        ],
        "diagnostic_plan_sha256": diagnosis.original.protocol.sha(out / "plan.json"),
        "scientific_validation_or_final_test_used": False,
        "closed_P10_resumed": False,
        "closed_P10_joint_gate_passed": False,
        "production_code_or_protocol_changed": False,
        "automatic_training_restart": False,
        "stage2_ready": False,
        "artifact_sha256": {str(p): diagnosis.original.protocol.sha(p) for p in files},
        "diagnostic_code_sha256": {str(p): diagnosis.original.protocol.sha(p) for p in scripts},
        "original_frozen_dependencies_checked": len(plan["dependencies"]),
    }
    diagnosis.original.protocol.freeze(out / "closeout.json", result)
    print(
        json.dumps(
            {
                "conclusion": result["conclusion"],
                "native_XPU_error_range": result["dataset_free_reproducer"][
                    "native_relative_L2_error_range"
                ],
                "candidate_error_max": max(composite_errors),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
