"""Read-only P10 prediction replay and completed-run audit; no fitting or tuning."""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import rclat_p10_train as study
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.diagnostics.rclat import text_errors
from bioacoustic_pipeline.representation.rclat_ctc import decode_logits
from bioacoustic_pipeline.representation.rclat_scale_v3 import ScaleCTC


def abstain():
    spec, _ = study.check()
    receipt = json.loads((study.OUT / "cache_integrity.json").read_text())
    for path, digest in receipt["files"].items():
        if study.protocol.sha(path) != digest:
            raise ValueError("Feature cache changed")
    arms = {}
    artifacts = {}
    for arm in ["mel", "combined"]:
        paths = {
            kind: study.OUT / f"{arm}_{kind}.json" for kind in ["STARTED", "scaler", "FAILURE"]
        }
        values = {kind: json.loads(path.read_text()) for kind, path in paths.items()}
        marker = values["STARTED"]
        if marker["preregistration_sha256"] != study.protocol.sha(
            study.OUT / "preregistration.json"
        ):
            raise ValueError("Started protocol mismatch")
        if marker["device"] != spec["frontends"][arm]["device"]:
            raise ValueError("Device assignment differs")
        epochs = sorted(study.OUT.glob(f"{arm}_epoch_*.json"))
        checkpoints = sorted(Path(f"checkpoints/rclat_p10/{arm}").glob("*.pt"))
        if epochs or checkpoints or (study.OUT / f"{arm}_result.json").exists():
            raise ValueError("This closeout requires the observed pre-first-epoch failure")
        if "Nonfinite parameter gradient" not in values["FAILURE"]["error"]:
            raise ValueError("Unexpected failure; do not misclassify")
        if values["FAILURE"]["automatic_retry_permitted"]:
            raise ValueError("Failure policy changed")
        log = study.OUT / f"{arm}_execution.log"
        paths["log"] = log
        artifacts.update({str(path): study.protocol.sha(path) for path in paths.values()})
        arms[arm] = {
            "device": marker["device"],
            "training_attempted": True,
            "completed_epochs": 0,
            "epoch_checkpoints": 0,
            "status": "HALTED_REGISTERED_NUMERICAL_ABSTENTION",
            "observed_failure": "Nonfinite return from clip_grad_norm_ after backward; runner raised ValueError('Nonfinite parameter gradient').",
            "guard_context": "Failing batch passed finite-logit and finite-CTC-loss checks. Individual-gradient finiteness versus norm overflow is not resolved by this logging.",
            "exact_failing_step": None,
            "completed_optimizer_updates": None,
            "validation_scored": False,
            "speaker_macro_cer": None,
            "nonempty_fraction": None,
            "learning_thresholds_assessable": False,
            "learning_gate_passed": False,
            "retries": 0,
        }
    study.protocol.freeze(
        study.OUT / "execution_closeout.json",
        {
            "status": "CLOSED_P10_NUMERICAL_ABSTENTION_NO_VALIDATION_RESULT",
            "closed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preregistration_sha256": study.protocol.sha(study.OUT / "preregistration.json"),
            "cache_integrity_sha256": study.protocol.sha(study.OUT / "cache_integrity.json"),
            "all_feature_cache_hashes_verified": len(receipt["files"]),
            "cache_preparation_wall_seconds": receipt["wall_seconds"],
            "arms": arms,
            "both_arms_completed_30_epochs": False,
            "joint_learning_gate_passed": False,
            "gate_interpretation": "Not passed because execution is incomplete under the registered abstention rule; no validation transcription-learning result is available.",
            "numerical_root_cause_established": False,
            "post_failure_fitting_or_policy_change": False,
            "final_test_scored": False,
            "automatic_100h_progression": False,
            "superiority_hypothesis_tested": False,
            "semantic_reversibility_established": False,
            "stage2_ready": False,
            "artifact_sha256": artifacts,
            "interpretation_limit": "This is a numerical execution outcome, not measured failure against the CER or nonempty-output thresholds. Engineering memorisation checks did not guarantee stability on the full P10 batches.",
        },
    )
    print(
        "Both arms closed under numerical abstention; no validation scores and no automatic retry."
    )


def replay(arm):
    spec, cohort = study.check()
    result = json.loads((study.OUT / f"{arm}_result.json").read_text())
    epoch = result["selected"]["epoch"]
    checkpoint = Path(f"checkpoints/rclat_p10/{arm}/epoch_{epoch:02d}.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["preregistration_sha256"] != study.protocol.sha(study.OUT / "preregistration.json"):
        raise ValueError("Checkpoint protocol differs")
    expected_steps = epoch * math.ceil(len(cohort["selected_training"]) / 4)
    if saved["epoch"] != epoch or saved["step"] != expected_steps:
        raise ValueError("Checkpoint training budget mismatch")
    device = spec["frontends"][arm]["device"]
    torch.xpu.set_device(device)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = ScaleCTC(
        spec["frontends"][arm]["dimensions"],
        width=256,
        layers=12,
        heads=4,
        expansion=1024,
        kernel=15,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    scaler = json.loads((study.OUT / f"{arm}_scaler.json").read_text())
    rows = sorted(cohort["selected_validation"], key=lambda r: r["uid"])
    stored = json.loads((study.OUT / f"{arm}_epoch_{epoch:02d}.json").read_text())
    predictions = {r["uid"]: r["hypothesis"] for r in stored["predictions"]}
    matched = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 4):
            batch = rows[start : start + 4]
            arrays = [frozen_scale(*study.sequence(r["uid"], arm), scaler) for r in batch]
            lengths = torch.tensor([len(x) for x, _ in arrays])
            values = pad_sequence([torch.from_numpy(x) for x, _ in arrays], batch_first=True).to(
                device
            )
            masks = pad_sequence([torch.from_numpy(m) for _, m in arrays], batch_first=True).to(
                device
            )
            logits, lengths = model(values, masks, lengths)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite replay logits")
            for row, logit, length in zip(batch, logits.cpu(), lengths):
                if decode_logits(logit, length) != predictions[row["uid"]]:
                    raise ValueError(f"Checkpoint replay differs: {row['uid']}")
                matched += 1
    study.protocol.freeze(
        study.OUT / f"{arm}_prediction_replay.json",
        {
            "arm": arm,
            "epoch": epoch,
            "device": device,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": study.protocol.sha(checkpoint),
            "matched_validation_predictions": matched,
            "all_matched": matched == len(rows),
            "fitting_performed": False,
            "final_test_used": False,
        },
    )
    print(f"{arm}: {matched} selected-checkpoint predictions replayed exactly", flush=True)


def audit():
    _, cohort = study.check()
    cache_receipt = json.loads((study.OUT / "cache_integrity.json").read_text())
    for path, digest in cache_receipt["files"].items():
        if study.protocol.sha(path) != digest:
            raise ValueError("Feature cache changed after fitting")
    references = {r["uid"]: r for r in cohort["selected_validation"]}
    summaries = {}
    checkpoints = {}
    all_speakers = []
    for arm in ["mel", "combined"]:
        result = json.loads((study.OUT / f"{arm}_result.json").read_text())
        marker = json.loads((study.OUT / f"{arm}_STARTED.json").read_text())
        if marker["preregistration_sha256"] != study.protocol.sha(
            study.OUT / "preregistration.json"
        ):
            raise ValueError("Start marker protocol mismatch")
        replay_result = json.loads((study.OUT / f"{arm}_prediction_replay.json").read_text())
        if not replay_result["all_matched"] or replay_result[
            "matched_validation_predictions"
        ] != len(references):
            raise ValueError("Missing full prediction replay")
        history = []
        for epoch in range(1, 31):
            path = study.OUT / f"{arm}_epoch_{epoch:02d}.json"
            saved = json.loads(path.read_text())
            rows = saved["predictions"]
            if len(rows) != len(references) or {r["uid"] for r in rows} != set(references):
                raise ValueError("Validation population changed")
            for row in rows:
                ref = references[row["uid"]]
                if row["speaker"] != ref["speaker"]:
                    raise ValueError("Speaker identity mismatch")
                metrics = text_errors(ref["transcript"], row["hypothesis"])
                if any(row[key] != value for key, value in metrics.items()):
                    raise ValueError("Saved transcript metrics mismatch")
            aggregate = study.aggregate(rows)
            for key in [
                "speaker_macro_cer",
                "speaker_macro_wer",
                "pooled_cer",
                "pooled_wer",
                "nonempty_fraction",
            ]:
                if not np.isclose(aggregate[key], saved[key], atol=1e-12, rtol=0):
                    raise ValueError("Saved aggregation mismatch")
            history.append({k: v for k, v in saved.items() if k != "predictions"})
            checkpoint = Path(f"checkpoints/rclat_p10/{arm}/epoch_{epoch:02d}.pt")
            checkpoints[str(checkpoint)] = study.protocol.sha(checkpoint)
        chosen = min(history, key=lambda r: (r["speaker_macro_cer"], r["epoch"]))
        if result["selected"] != chosen or result["final"] != history[-1]:
            raise ValueError("Checkpoint selection differs from registered rule")
        passed = chosen["speaker_macro_cer"] < 0.5 and chosen["nonempty_fraction"] >= 0.95
        if passed != result["learning_gate_passed"]:
            raise ValueError("Learning gate mismatch")
        selected_hash = checkpoints[replay_result["checkpoint"]]
        if selected_hash != replay_result["checkpoint_sha256"]:
            raise ValueError("Replayed checkpoint changed")
        summaries[arm] = {
            "selected_epoch": chosen["epoch"],
            "speaker_macro_cer": chosen["speaker_macro_cer"],
            "speaker_macro_wer": chosen["speaker_macro_wer"],
            "pooled_cer": chosen["pooled_cer"],
            "pooled_wer": chosen["pooled_wer"],
            "nonempty_fraction": chosen["nonempty_fraction"],
            "learning_gate_passed": passed,
            "final_epoch": history[-1],
            "epoch_history": history,
        }
        all_speakers.extend(
            {"arm": arm, "selected_epoch": chosen["epoch"], **r} for r in chosen["speaker_results"]
        )
    with (study.OUT / "speaker_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["arm", "selected_epoch", "speaker", "cer", "wer"]
        )
        writer.writeheader()
        writer.writerows(all_speakers)
    study.protocol.freeze(
        study.OUT / "execution_closeout.json",
        {
            "status": "COMPLETED_FROZEN_P10_DEVELOPMENT_LEARNING_VALIDATION",
            "preregistration_sha256": study.protocol.sha(study.OUT / "preregistration.json"),
            "cache_integrity_sha256": study.protocol.sha(study.OUT / "cache_integrity.json"),
            "both_arms_completed_30_epochs": True,
            "both_prediction_replays_exact": True,
            "arms": summaries,
            "joint_learning_gate_passed": all(
                r["learning_gate_passed"] for r in summaries.values()
            ),
            "all_60_epoch_metrics_recomputed": True,
            "checkpoint_sha256": checkpoints,
            "superiority_hypothesis_tested": False,
            "final_test_scored": False,
            "post_evaluation_tuning_performed": False,
            "automatic_100h_progression": False,
            "semantic_reversibility_established": False,
            "stage2_ready": False,
        },
    )
    print(
        json.dumps(
            {
                arm: {k: v for k, v in summary.items() if k not in ["epoch_history", "final_epoch"]}
                for arm, summary in summaries.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["replay", "audit", "abstain"])
    parser.add_argument("--arm", choices=["mel", "combined"])
    args = parser.parse_args()
    if args.action == "replay":
        if args.arm is None:
            parser.error("replay requires --arm")
        replay(args.arm)
    elif args.action == "audit":
        audit()
    else:
        abstain()
