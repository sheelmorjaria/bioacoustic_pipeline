"""Freeze a new baseline-only 10-hour learning-validation protocol; no training."""

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("outputs/rclat_bilstm_p10_v1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_contract(p):
    assert p["training"]["epochs"] == 30 and p["training"]["updates"] == 21120
    assert p["training"]["batch_utterances"] == 4
    assert p["training"]["learning_rate"] == "min(original_rate(step,21120),0.0005005)"
    assert p["learning_gate"]["speaker_macro_CER_strict_maximum"] == 0.5
    assert p["learning_gate"]["nonempty_minimum"] == 0.95
    assert p["checkpoint_policy"]["ties"] == "earliest_epoch"
    assert p["checkpoint_policy"]["report_final_epoch"]
    assert not p["automatic_scientific_start"]
    assert not p["P100_authorized"] and not p["P960_authorized"] and not p["stage2_ready"]
    assert p["final_test_access"] == "prohibited"
    assert p["training"]["fresh_initialisation"] and not p["training"]["resume"]


def main():
    root = ROOT
    root.mkdir(exist_ok=False)
    prior_path = Path("outputs/rclat_p10_corrected_v3/preregistration.json")
    prior = json.loads(prior_path.read_text())
    source = Path("outputs/rclat_p10/cohort.json")
    cohort = json.loads(source.read_text())
    training = cohort["selected_training"]
    validation = cohort["selected_validation"]
    assert len(training) == 2814 and len(validation) == 2864
    assert len({r["speaker"] for r in training}) == 251
    assert len({r["speaker"] for r in validation}) == 33
    assert not {r["speaker"] for r in training} & {r["speaker"] for r in validation}
    assert not {r["uid"] for r in training} & {r["uid"] for r in validation}
    assert not {r["decoded_pcm16_sha256"] for r in training} & {
        r["decoded_pcm16_sha256"] for r in validation
    }
    integrity = Path("outputs/rclat_p10/cache_integrity.json")
    files = json.loads(integrity.read_text())["files"]
    assert len(files) == 5678
    for row in training + validation:
        assert row["ctc_frames"] >= row["ctc_minimum_frames"]
        assert str(Path("cache/rclat_p10") / (row["uid"] + ".npz")) in files
    previous = Path("outputs/rclat_simple_ctc_diagnostic_v2/summary.json")
    passed = json.loads(previous.read_text())
    assert all(
        v["result"]["completed"] and v["result"]["training_fit_indicator_passed"]
        for v in passed["cases"].values()
    )
    dependencies = {
        str(prior_path): sha(prior_path),
        str(source): sha(source),
        str(integrity): sha(integrity),
        str(previous): sha(previous),
    }
    for path in [
        "outputs/rclat_p10/mel_scaler.json",
        "outputs/rclat_simple_ctc_diagnostic_v2/initial_state_reference.json",
        "outputs/rclat_simple_ctc_diagnostic_v2/execution_closeout.json",
        "outputs/rclat_bilstm_matched_grounding_v1/closeout.json",
        "outputs/rclat_grounding_source_support_v1/selected_metadata.json",
        "scripts/rclat_bilstm_p10_protocol.py",
        "tests/test_rclat_bilstm_p10_protocol.py",
        "src/bioacoustic_pipeline/representation/rclat_simple_ctc.py",
        "scripts/rclat_simple_ctc_diagnostic_v2.py",
        "scripts/rclat_ctc_lr_diagnostic.py",
        "scripts/rclat_p10_train.py",
        "scripts/rclat_p10_c3_train.py",
        "src/bioacoustic_pipeline/representation/rclat_ctc.py",
        "src/bioacoustic_pipeline/diagnostics/frozen_acoustic_transfer.py",
    ]:
        dependencies[path] = sha(path)
    protocol = {
        "study": "RCLAT-BiLSTM-P10-v1",
        "status": "LOCALLY_FROZEN_DESIGN_NOT_EXTERNAL_REGISTRATION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "New 10-hour log-mel baseline cross-speaker learning validation; not representation superiority and not a modification of closed P10-C3.",
        "population": {
            "training_utterances": 2814,
            "training_speakers": 251,
            "training_hours": sum(r["samples"] for r in training) / 16000 / 3600,
            "validation_utterances": 2864,
            "validation_speakers": 33,
            "validation_hours": sum(r["samples"] for r in validation) / 16000 / 3600,
            "selection": "Reuse exact P10 deterministic populations and ordering; no adaptive exclusions.",
            "validation_exposure": "Previously exposed dev-other development population; not untouched confirmatory testing.",
        },
        "architecture": {
            "module": "bioacoustic_pipeline.representation.rclat_simple_ctc.SimpleCTC",
            "parameters": 781725,
            "input": "64D log-mel plus original validity masks",
            "projection_width": 128,
            "subsampling": "two128-channel kernel3 stride2 bias-free convolutions with SiLU; ceil(native_length/4)",
            "recurrent": "two bidirectional LSTM layers, hidden128 each direction; separate valid-length sequences to avoid backward recurrence seeing padding",
            "output": "linear256-to29; no dropout, no positional encoding",
        },
        "features": "Unchanged existing source feature caches and training-only mel scaler; no augmentation, denoising, external language model, pretrained weights or supplied word boundaries.",
        "training": {
            "epochs": 30,
            "updates": 21120,
            "batch_utterances": 4,
            "batch_order": prior["training"]["order"],
            "learning_rate": "min(original_rate(step,21120),0.0005005)",
            "optimizer": prior["training"]["optimizer"],
            "gradient_clip_norm": 5.0,
            "precision": "FP32; no autocast or scaling",
            "ctc": prior["training"]["ctc"],
            "fresh_initialisation": True,
            "seed": 0,
            "resume": False,
            "early_stopping": False,
            "internal_walltime_seconds_per_device": 43200,
            "external_walltime_seconds_per_device": 43320,
        },
        "devices": {
            "xpu:0": "primary scientific run",
            "xpu:1": "same-seed computational reproduction, not independent statistical replication",
        },
        "checkpoint_policy": {
            "evaluate": "all2864validation utterances after each completed epoch",
            "save": "all30epoch model,optimizer,RNG,step,provenance checkpoints; per-utterance predictions and S/D/I at every epoch",
            "select": "minimum validation speaker-macro CER after completing all30epochs",
            "ties": "earliest_epoch",
            "report_final_epoch": True,
            "no_intermediate_success_stop": True,
        },
        "learning_gate": {
            "speaker_macro_CER_strict_maximum": 0.5,
            "nonempty_minimum": 0.95,
            "apply_to": "selected checkpoint per device; both devices must complete and satisfy both criteria",
            "incomplete_execution": "not a completed learning failure; gate unpassed/unevaluated; preserve artifacts",
            "interpretation": "Developmental learning validation only; no confirmatory accuracy, semantic or representation-superiority inference.",
        },
        "secondary_grounding": {
            "inventory": "outputs/rclat_grounding_source_support_v1/selected_metadata.json",
            "size": 384,
            "speakers": 24,
            "pairs": 192,
            "when": "after full training and frozen checkpoint selection; same selected checkpoint as primary",
            "endpoint": "speaker-macro native-mismatched minus matched CER; paired speaker-cluster percentile95 interval",
            "bootstrap_seed": 20261022,
            "bootstrap_replicates": 10000,
            "method": "sample24speaker IDs with replacement, preserve all8pairs per speaker; linear percentile quantiles",
            "limitations": "Overlaps checkpoint-selection validation population; exposed development evidence, not independent confirmation or a substitute for the primary gate.",
            "selection_use": "prohibited",
        },
        "runtime": prior["runtime_policy"],
        "numerical_guards": prior["training"]["finite_guards"],
        "logging": "Per-step epoch/step/UIDs/loss/LR/gradient norm/finite guards/hostRSS/nativeheap/XPUmemory; validation boundaries and checkpoint hashes; no silent omission.",
        "execution_prerequisites": [
            "Implement and hash-pin a dedicated runner; protocol freeze alone does not start training.",
            "Fresh CPU–XPU parity on both devices for fixed first, shortest and longest training batches including CTC gradients, global relative tolerance1e-4.",
            "Actual source/cache hash and all-population CTC-length verification before updates.",
            "Fresh seed0 digest must match the successful small-encoder initial-state reference; no trained checkpoint load.",
            "Retain exact model and optimisation; any failure correction requires a separately versioned protocol.",
        ],
        "failure_policy": "No automatic retry,resume,LR/batch/precision/cache changes,budget escalation or outcome-driven repair. Preserve exact failing step and inputs; no final gate if either execution is incomplete.",
        "final_test_access": "prohibited",
        "automatic_scientific_start": False,
        "runner_implemented": False,
        "scientific_training_started": False,
        "P10_C3_reopened": False,
        "P100_authorized": False,
        "P960_authorized": False,
        "stage2_ready": False,
        "dependencies": dependencies,
    }
    validate_contract(protocol)
    (root / "preregistration.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (root / "population_check.json").write_text(
        json.dumps(
            {
                "training_utterances": len(training),
                "validation_utterances": len(validation),
                "speaker_overlap": False,
                "decoded_audio_overlap": False,
                "cache_entries_present_in_frozen_integrity_inventory": 5678,
                "recorded_CTC_feasibility": True,
                "actual_cache_rehash": "pending required execution preflight",
                "no_inference_or_training": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("Locally frozen protocol SHA256", sha(root / "preregistration.json"))


if __name__ == "__main__":
    main()
