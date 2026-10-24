"""Dedicated frozen 10-hour BiLSTM runner; fresh starts and no repairs."""

import argparse
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import rclat_simple_ctc_diagnostic_v2 as core
from rclat_ctc_lr_diagnostic import learning_rate
from rclat_bilstm_native_grounding import pairing_scores
from rclat_bilstm_matched_summary import paired_interval
from bioacoustic_pipeline.diagnostics.rclat import normalize_text

from bioacoustic_pipeline.representation.rclat_dropout_ctc import DropoutCTC

DROPOUT = 0.0

ROOT = Path("outputs/rclat_bilstm_dropout_v1")
CHECKPOINTS = Path("checkpoints/rclat_bilstm_dropout_v1")


def check():
    spec = json.loads((ROOT / "preregistration.json").read_text())
    execution = json.loads((ROOT / "execution_manifest.json").read_text())
    assert execution["protocol_sha256"] == core.sha(ROOT / "preregistration.json")
    for mapping in [spec["dependencies"], execution["dependencies"]]:
        for p, digest in mapping.items():
            if core.sha(p) != digest:
                raise ValueError("Changed dependency: " + p)
    return spec, json.loads(Path("outputs/rclat_p10/cohort.json").read_text())


def verify_inputs():
    _, cohort = check()
    inventory = json.loads(Path("outputs/rclat_p10/cache_integrity.json").read_text())["files"]
    records = {}
    for row in cohort["selected_training"] + cohort["selected_validation"]:
        path = Path("cache/rclat_p10") / (row["uid"] + ".npz")
        if core.sha(path) != inventory[str(path)] or core.sha(row["path"]) != row["source_sha256"]:
            raise ValueError("Source/cache checksum mismatch: " + row["uid"])
        with np.load(path, allow_pickle=False) as f:
            n = len(f["mel"])
            assert f["mel"].shape == (n, 64) and f["mel_valid"].shape == (n, 64)
        target = core.encode_text(row["transcript"])
        minimum = len(target) + int((target[1:] == target[:-1]).sum())
        assert (n + 3) // 4 >= minimum
        records[row["uid"]] = {
            "native_frames": n,
            "target_length": len(target),
            "minimum_CTC_frames": minimum,
        }
    assert len(records) == 5678
    receipt = {
        "passed": True,
        "source_and_cache_files_verified": 5678,
        "actual_CTC_lengths_verified": 5678,
        "records": records,
        "protocol_sha256": core.sha(ROOT / "preregistration.json"),
        "execution_sha256": core.sha(ROOT / "execution_manifest.json"),
        "training_updates": 0,
    }
    core.write(ROOT / "input_verification.json", receipt)
    print("All5678 sources/cache entries and actual CTC lengths verified.", flush=True)


def setup(device):
    core.predecessor.check()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    torch.xpu.set_device(device)


def batcher(device):
    scaler = json.loads(Path("outputs/rclat_p10/mel_scaler.json").read_text())

    def batch(rows, where=device):
        arrays = [
            core.frozen_scale(*core.predecessor.original.sequence(r["uid"], "mel"), scaler)
            for r in rows
        ]
        inputs = (
            pad_sequence([torch.from_numpy(x) for x, m in arrays], batch_first=True).to(where),
            pad_sequence([torch.from_numpy(m) for x, m in arrays], batch_first=True).to(where),
            torch.tensor([len(x) for x, m in arrays]),
        )
        return inputs, [core.encode_text(r["transcript"]) for r in rows]

    return batch


def preflight(device):
    _, cohort = check()
    receipt = json.loads((ROOT / "input_verification.json").read_text())
    assert receipt["passed"] and receipt["execution_sha256"] == core.sha(
        ROOT / "execution_manifest.json"
    )
    setup(device)
    out = ROOT / device.replace(":", "_")
    core.write(
        out / "preflight_STARTED.json",
        {"device": device, "execution_sha256": core.sha(ROOT / "execution_manifest.json")},
    )
    rows = cohort["selected_training"]
    cases = {
        "first": core.predecessor.original.epoch_order(rows, 1)[:4],
        "shortest": sorted(
            rows, key=lambda r: (receipt["records"][r["uid"]]["native_frames"], r["uid"])
        )[:4],
        "longest": sorted(
            rows, key=lambda r: (-receipt["records"][r["uid"]]["native_frames"], r["uid"])
        )[:4],
    }
    core.OUT = out
    core.build = lambda: DropoutCTC(0.0)
    for name, group in cases.items():
        core.preflight(
            device, {"parity_tolerance": 1e-4}, {"training": group}, batcher(device), name
        )
    # The random masks differ across devices: validate each realised mask analytically.
    x = torch.ones(65536, device=device, requires_grad=True)
    y = torch.nn.functional.dropout(x, p=0.1, training=True)
    y.sum().backward()
    expected = (y.detach().cpu() != 0).float() / 0.9
    error = float((x.grad.detach().cpu() - expected).abs().max())
    if error > 1e-6:
        raise FloatingPointError("Dropout gradient versus observed-mask reference failed")
    model = DropoutCTC(DROPOUT).to(device).train()
    inputs, targets = batcher(device)(cases["first"])
    initial = core.predecessor.initial_digest(model)
    logits, lengths = model(*inputs)
    core.assert_lengths(logits, lengths, inputs[2], targets)
    loss = core.ctc_objective(logits, lengths, targets)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Training-mode dropout CTC loss nonfinite")
    loss.backward()
    core.check_gradients(model)
    assert core.predecessor.initial_digest(model) == initial
    core.write(
        out / "dropout_path_result.json",
        {
            "passed": True,
            "dropout": DROPOUT,
            "observed_mask_gradient_max_abs_error": error,
            "optimizer_updates": 0,
            "deterministic_parity_uses_dropout_disabled": True,
        },
    )
    core.build = lambda: DropoutCTC(DROPOUT)
    core.write(
        out / "preflight_result.json",
        {
            "passed": True,
            "cases": list(cases),
            "training_updates": 0,
            "execution_sha256": core.sha(ROOT / "execution_manifest.json"),
        },
    )
    print(device, "three numerical preflights passed", flush=True)


def select_checkpoint(history):
    if len(history) != 30 or {r["epoch"] for r in history} != set(range(1, 31)):
        raise ValueError("Full30epoch history required")
    return min(history, key=lambda r: (r["speaker_macro_cer"], r["epoch"]))


def learning_gate(metrics):
    return metrics["speaker_macro_cer"] < 0.5 and metrics["nonempty_fraction"] >= 0.95


def score(row, hypothesis, native_length):
    reference = normalize_text(row["transcript"])
    hypothesis = normalize_text(hypothesis)
    c = core.edit_counts(reference.replace(" ", ""), hypothesis.replace(" ", ""))
    w = core.edit_counts(reference.split(), hypothesis.split())
    return {
        "uid": row["uid"],
        "speaker": row["speaker"],
        "reference": reference,
        "hypothesis": hypothesis,
        "native_length": int(native_length),
        "reference_characters": len(reference.replace(" ", "")),
        "reference_words": len(reference.split()),
        "character_errors": c["errors"],
        "word_errors": w["errors"],
        "character_decomposition": c,
        "word_decomposition": w,
        "cer": c["errors"] / len(reference.replace(" ", "")),
        "wer": w["errors"] / len(reference.split()),
    }


def train(device):
    spec, cohort = check()
    receipt = json.loads((ROOT / "input_verification.json").read_text())
    assert receipt["passed"] and receipt["execution_sha256"] == core.sha(
        ROOT / "execution_manifest.json"
    )
    for d in ["xpu_0", "xpu_1"]:
        p = json.loads((ROOT / d / "preflight_result.json").read_text())
        assert p["passed"] and p["execution_sha256"] == core.sha(ROOT / "execution_manifest.json")
    setup(device)
    out = ROOT / device.replace(":", "_")
    core.write(
        out / "STARTED.json",
        {
            "device": device,
            "protocol_sha256": core.sha(ROOT / "preregistration.json"),
            "execution_sha256": core.sha(ROOT / "execution_manifest.json"),
            "fresh_initialisation": True,
            "checkpoint_loaded": False,
            "dropout": DROPOUT,
            "software": core.predecessor.original.protocol.software_versions(),
        },
    )
    model = core.build()
    digest = core.predecessor.initial_digest(model)
    assert (
        digest
        == json.loads(
            Path("outputs/rclat_simple_ctc_diagnostic_v2/initial_state_reference.json").read_text()
        )["full_state_sha256"]
    )
    core.write(out / "initial_state.json", {"full_state_sha256": digest, "seed": 0, "fresh": True})
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    core.check_parameters_and_optimizer(model, optimizer)
    batch = batcher(device)
    validation = sorted(cohort["selected_validation"], key=lambda r: r["uid"])
    step = 0
    history = []
    start = time.monotonic()

    def bound():
        if time.monotonic() - start > spec["training"]["internal_walltime_seconds_per_device"]:
            raise TimeoutError("Frozen12hour execution bound")

    with (out / "steps.jsonl").open("x", buffering=1) as journal:

        def event(kind, **values):
            journal.write(json.dumps({"event": kind, "step": step, **values}) + "\n")

        for epoch in range(1, 31):
            ordered = core.predecessor.original.epoch_order(cohort["selected_training"], epoch)
            losses = []
            for pos in range(0, len(ordered), 4):
                bound()
                step += 1
                model.train()
                rows = ordered[pos : pos + 4]
                rate = learning_rate(step, "capped250")
                event("begin", epoch=epoch, uids=[r["uid"] for r in rows], learning_rate=rate)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                inputs, targets = batch(rows)
                event(
                    "forward_begin",
                    epoch=epoch,
                    shape=list(inputs[0].shape),
                    native_lengths=inputs[2].tolist(),
                )
                logits, lengths = model(*inputs)
                core.assert_lengths(logits, lengths, inputs[2], targets)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite logits")
                loss = core.ctc_objective(logits, lengths, targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                gradients = core.check_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
                checks = core.check_parameters_and_optimizer(model, optimizer)
                losses.append(float(loss.detach()))
                event(
                    "step_complete",
                    epoch=epoch,
                    loss=losses[-1],
                    gradient_norm=float(norm),
                    learning_rate=rate,
                    gradients_checked=gradients,
                    **checks,
                    **core.predecessor.telemetry(device),
                )
                del inputs, targets, logits, lengths, loss
            event("validation_begin", epoch=epoch, **core.predecessor.telemetry(device))
            model.eval()
            predictions = []
            with torch.no_grad():
                for pos in range(0, len(validation), 4):
                    bound()
                    rows = validation[pos : pos + 4]
                    inputs, targets = batch(rows)
                    logits, lengths = model(*inputs)
                    core.assert_lengths(logits, lengths, inputs[2], targets)
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Nonfinite validation logits")
                    for row, x, n, native in zip(rows, logits.cpu(), lengths, inputs[2]):
                        predictions.append(score(row, core.decode_logits(x, int(n)), native))
                    event("validation_batch", epoch=epoch, batch=pos // 4 + 1)
                    del inputs, targets, logits, lengths
            metrics = core.predecessor.original.aggregate(predictions)
            item = {
                "epoch": epoch,
                "step": step,
                "mean_batch_training_loss": float(np.mean(losses)),
                **metrics,
            }
            history.append(item)
            path = CHECKPOINTS / device.replace(":", "_") / f"epoch_{epoch:02d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise RuntimeError("Existing checkpoint; overwrite prohibited")
            rng = np.random.get_state()
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "torch_rng": torch.get_rng_state(),
                    "xpu_rng": torch.xpu.get_rng_state_all(),
                    "python_rng": random.getstate(),
                    "numpy_rng": [rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]],
                    "protocol_sha256": core.sha(ROOT / "preregistration.json"),
                    "execution_sha256": core.sha(ROOT / "execution_manifest.json"),
                    "resume_permitted": False,
                },
                path,
            )
            core.write(
                out / f"epoch_{epoch:02d}.json",
                {
                    **item,
                    "predictions": predictions,
                    "checkpoint": str(path),
                    "checkpoint_sha256": core.sha(path),
                },
            )
            event("epoch_complete", epoch=epoch, **core.predecessor.telemetry(device))
            print(json.dumps({"device": device, **item}), flush=True)
        assert step == 21120
        selected = select_checkpoint(history)
        records = json.loads((out / f"epoch_{selected['epoch']:02d}.json").read_text())[
            "predictions"
        ]
        inventory = json.loads(Path(spec["secondary_grounding"]["inventory"]).read_text())
        scores = pairing_scores(
            [r for r in records if r["uid"] in inventory["pairing"]], inventory["pairing"]
        )
        matched = core.predecessor.original.aggregate(scores["matched"])
        wrong = core.predecessor.original.aggregate(scores["native_swapped"])
        delta = [
            r["cer"]
            - next(x["cer"] for x in matched["speaker_results"] if x["speaker"] == r["speaker"])
            for r in wrong["speaker_results"]
        ]
        grounding = {
            "matched": matched,
            "native_swapped": wrong,
            "interval": paired_interval(delta, 20261022, 10000),
            "scores": scores,
        }
        core.write(out / "secondary_grounding.json", grounding)
        training_metrics = {}
        for evaluated_epoch in sorted({selected["epoch"], 30}):
            checkpoint_path = (
                CHECKPOINTS / device.replace(":", "_") / f"epoch_{evaluated_epoch:02d}.pt"
            )
            saved = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            model.load_state_dict(saved["model"], strict=True)
            del saved
            model.eval()
            training_predictions = []
            ordered_training = sorted(cohort["selected_training"], key=lambda r: r["uid"])
            with torch.no_grad():
                for pos in range(0, len(ordered_training), 4):
                    bound()
                    rows = ordered_training[pos : pos + 4]
                    inputs, targets = batch(rows)
                    logits, lengths = model(*inputs)
                    core.assert_lengths(logits, lengths, inputs[2], targets)
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Post-training inference logits nonfinite")
                    for row, x, n, native in zip(rows, logits.cpu(), lengths, inputs[2]):
                        training_predictions.append(
                            score(row, core.decode_logits(x, int(n)), native)
                        )
                    del inputs, targets, logits, lengths
            training_metrics[str(evaluated_epoch)] = {
                "metrics": core.predecessor.original.aggregate(training_predictions),
                "predictions": training_predictions,
                "optimizer_updates": 0,
            }
        core.write(out / "training_metrics.json", training_metrics)
        core.write(
            out / "result.json",
            {
                "completed": True,
                "epochs": 30,
                "updates": step,
                "selected": selected,
                "final": history[-1],
                "learning_gate_passed": learning_gate(selected),
                "elapsed_seconds": time.monotonic() - start,
                "P100_authorized": False,
                "P960_authorized": False,
                "stage2_ready": False,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["verify", "preflight", "train"])
    parser.add_argument("--device", choices=["xpu:0", "xpu:1"])
    args = parser.parse_args()
    if args.device is not None:
        DROPOUT = 0.0 if args.device == "xpu:0" else 0.1
    core.build = lambda: DropoutCTC(DROPOUT)
    try:
        if args.action == "verify":
            verify_inputs()
        elif args.device is None:
            parser.error("--device required")
        elif args.action == "preflight":
            preflight(args.device)
        else:
            train(args.device)
    except Exception:
        directory = ROOT / (args.device.replace(":", "_") if args.device else "verification")
        path = directory / (args.action + "_FAILURE.json")
        if not path.exists():
            core.write(
                path,
                {
                    "traceback": traceback.format_exc(),
                    "automatic_retry": False,
                    "P100_authorized": False,
                    "stage2_ready": False,
                },
            )
        raise
