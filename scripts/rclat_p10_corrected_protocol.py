"""Freeze a distinct corrected execution after bounded version4 engineering checks."""

import copy
import json
from pathlib import Path
import time

import rclat_p10_corrected_preflight as engineering
import rclat_p10_corrected_train as corrected


def main():
    original, cohort, plan = engineering.check()
    if (corrected.OUT / "preregistration.json").exists():
        raise RuntimeError("Corrected execution already frozen")
    results = {}
    for arm, device in plan["devices"].items():
        path = engineering.OUT / f"{arm}_results.json"
        data = json.loads(path.read_text())
        if (
            data["device"] != device
            or not data["engineering_gate_passed"]
            or data["repeated_optimizer_updates"] != 200
        ):
            raise ValueError("Incomplete or mismatched engineering run")
        if data["plan_sha256"] != engineering.prior.original.protocol.sha(
            engineering.OUT / "plan.json"
        ):
            raise ValueError("Engineering plan identity changed")
        if len(data["synthetic_parity"]) != len(plan["synthetic_shapes_batch_length"]) or len(
            data["full_model_parity"]
        ) != len(plan["full_model_cases"]):
            raise ValueError("Incomplete parity coverage")
        if [r["step"] for r in data["step_trace"]] != list(range(1, 201)):
            raise ValueError("Engineering step trace incomplete")
        if not data["all_gradient_parameter_optimizer_state_checks_passed"]:
            raise ValueError("Finite-state engineering check failed")
        for row in data["step_trace"]:
            if (
                row["gradient_tensors_checked"] <= 0
                or row["parameters_checked"] <= 0
                or row["optimizer_tensors_checked"] <= 0
            ):
                raise ValueError("Missing per-step tensor checks")
        results[arm] = {
            "path": str(path),
            "sha256": engineering.prior.original.protocol.sha(path),
            "device": device,
            "updates": 200,
            "synthetic_cases": len(data["synthetic_parity"]),
            "full_model_cases": len(data["full_model_parity"]),
            "max_synthetic_gradient_relative_L2": max(
                r["gradient_relative_L2"] for r in data["synthetic_parity"]
            ),
            "max_model_gradient_relative_L2": max(
                r["gradient_relative_L2_error"] for r in data["full_model_parity"]
            ),
            "max_norm_relative_error": max(
                r["norm_relative_error"]
                for r in data["step_trace"]
                if r["norm_relative_error"] is not None
            ),
        }
    freeze = engineering.prior.original.protocol.freeze
    sha = engineering.prior.original.protocol.sha
    closeout = engineering.OUT / "closeout.json"
    freeze(
        closeout,
        {
            "status": "COMPLETED_VERSION4_ENGINEERING_VALIDATION",
            "plan_sha256": sha(engineering.OUT / "plan.json"),
            "arms": results,
            "joint_engineering_gate_passed": True,
            "all_gradient_parameter_optimizer_checks_passed": True,
            "scientific_validation_or_test_scored": False,
            "closed_P10_resumed": False,
            "interpretation": "Bounded numerical readiness,not30epoch stability or transcription learning.",
            "source_code_sha256": plan["dependencies"],
            "stage2_ready": False,
        },
    )
    spec = copy.deepcopy(original)
    spec["study"] = "RCLAT P10-C1: corrected composite-softmax execution v1"
    spec["status"] = "LOCALLY_FROZEN_CORRECTED_EXECUTION_NOT_EXTERNAL_REGISTRATION"
    spec["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    spec["predecessor"] = {
        "protocol": "outputs/rclat_p10/preregistration.json",
        "protocol_sha256": sha(Path("outputs/rclat_p10/preregistration.json")),
        "execution_closeout": "outputs/rclat_p10/execution_closeout.json",
        "status": "Closed numerical abstention; original joint learning gate remains not passed.",
        "diagnostic_closeout": "outputs/rclat_p10_numerical/closeout.json",
        "continuation_or_resume": False,
    }
    spec["purpose"] = (
        "New corrected numerical execution of the same P10 development learning comparison;not a continuation,superiority test or final-test confirmation."
    )
    spec["architecture"]["module"] = "rclat_scale_v4.ScaleCTC"
    spec["architecture"]["attention"] = (
        "Explicit FP32 attention with stable composite max/exp/sum softmax,finite padding bias;offline bidirectional. Same parameter schema,initialization and architecture as v3."
    )
    spec["features"] = spec["features"].replace(
        "Fit each scaler on selected training sources only.",
        "Reuse archived scalers fitted on the unchanged selected training sources only.",
    )
    spec["scaler_policy"] = (
        "Bit-exact reuse of original training-only mel/combined scalers; no validation fitting,renormalisation or feature-cache changes."
    )
    spec["training"]["finite_guards"] = (
        "Every step: finite logits/loss,individual gradients before clipping,finite clip norm,parameters and optimizer state after update. Failure names and exact epoch/step/UID journal preserved. No sanitisation or zeroing of bad gradients."
    )
    spec["training"]["resume"] = (
        "Not implemented. Existing start marker prohibits retry/resume; no closed-attempt weights or engineering states loaded."
    )
    spec["initialization"] += (
        " Fresh seed0 construction; engineering weights and optimizer state are never loaded."
    )
    spec["journal"] = (
        "Append begin,backward_complete,optimizer_begin,step_complete events for each batch. Save initial parameter hashes and common encoder/head hash;all30epoch checkpoints and prediction artifacts."
    )
    spec["source_population"] = {
        "cohort": "outputs/rclat_p10/cohort.json",
        "cohort_sha256": sha(Path("outputs/rclat_p10/cohort.json")),
        "training_utterances": len(cohort["selected_training"]),
        "validation_utterances": len(cohort["selected_validation"]),
        "shared_cache": "cache/rclat_p10",
        "cache_receipt": "outputs/rclat_p10/cache_integrity.json",
        "training_validation_reassignment": False,
    }
    spec["engineering_results"] = {arm: record["path"] for arm, record in results.items()}
    spec["engineering_interpretation"] = (
        "12synthetic and5fullmodel parity cases perarm plus200training-only updates perarm passed prospectively fixed tolerances. Does not establish full-run numerical stability or speech performance."
    )
    spec["additional_development_exposure"] = (
        "Engineering used first200original training batches and5metadata-selected training parity batches. No validation/final-test model inputs. All scientific runs start from fresh seed0; no engineering parameters reused."
    )
    spec["scientific_policy_changes"] = {
        "attention_softmax_implementation": "v3native replaced by mathematically equivalent v4composite",
        "added_guards_and_logging": True,
        "data_frontends_targets_horizons_batch_order_optimizer_budget_validation_selection_learning_thresholds_changed": False,
    }
    spec["abstention"] += (
        " Individual nonfinite gradients,parameters or optimizer tensors also halt the corrected arm;all failure context retained. No repair/retry within this protocol."
    )
    spec["scientific_training_started"] = False
    spec["training_started"] = False
    spec["automatic_100h_progression"] = False
    # Preserve original source dependencies and separately pin all changed code/evidence.
    paths = [
        Path(__file__),
        Path("scripts/rclat_p10_corrected_train.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_scale_v4.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_numerics.py"),
        Path("tests/test_rclat_scale_v4.py"),
        Path("outputs/rclat_p10/preregistration.json"),
        Path("outputs/rclat_p10/execution_closeout.json"),
        Path("outputs/rclat_p10/cache_integrity.json"),
        Path("outputs/rclat_p10/mel_scaler.json"),
        Path("outputs/rclat_p10/combined_scaler.json"),
        engineering.OUT / "plan.json",
        closeout,
        engineering.OUT / "mel_results.json",
        engineering.OUT / "combined_results.json",
    ]
    spec["dependencies"].update({str(p): sha(p) for p in paths})
    for key in ["selection", "frontends", "validation", "learning_gate", "software"]:
        if spec[key] != original[key]:
            raise ValueError(f"Original scientific rule changed: {key}")
    for key in [
        "epochs",
        "batch_utterances",
        "order",
        "optimizer",
        "learning_rate",
        "gradient_clip_norm",
        "precision",
        "ctc",
        "early_stopping",
    ]:
        if spec["training"][key] != original["training"][key]:
            raise ValueError(f"Original training policy changed: {key}")
    freeze(corrected.OUT / "preregistration.json", spec)
    print("P10-C1 separately frozen; no corrected scientific training started.")
    print("Corrected protocol SHA256:", sha(corrected.OUT / "preregistration.json"))


if __name__ == "__main__":
    main()
