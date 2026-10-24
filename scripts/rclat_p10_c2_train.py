"""P10-C2 fresh epoch-0 execution, with additional runtime telemetry.

No C1 checkpoint continuation, backend repair claim or automatic retry.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import rclat_p10_train as original
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.diagnostics.rclat import text_errors
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, decode_logits, encode_text
from bioacoustic_pipeline.representation.rclat_numerics import (
    check_gradients,
    check_parameters_and_optimizer,
)
from bioacoustic_pipeline.representation.rclat_scale_v4 import ScaleCTC
from rclat_p10_c1_shape_replay import telemetry

OUT = Path("outputs/rclat_p10_corrected_v2")
CHECKPOINTS = Path("checkpoints/rclat_p10_corrected_v2")


def check():
    original.check()
    spec = json.loads((OUT / "preregistration.json").read_text())
    for path, digest in spec["dependencies"].items():
        if original.protocol.sha(path) != digest:
            raise ValueError(f"Corrected execution dependency changed: {path}")
    if original.protocol.software_versions() != spec["software"]:
        raise ValueError("Corrected execution software versions changed")
    cohort = json.loads((original.OUT / "cohort.json").read_text())
    for arm, path in spec["engineering_results"].items():
        record = json.loads(Path(path).read_text())
        if not record["engineering_gate_passed"] or record["repeated_optimizer_updates"] != 200:
            raise ValueError(f"Incomplete engineering validation: {arm}")
    return spec, cohort


def verify_cache():
    spec, cohort = check()
    receipt = json.loads((original.OUT / "cache_integrity.json").read_text())
    expected = {
        str(original.CACHE / f"{r['uid']}.npz")
        for r in cohort["selected_training"] + cohort["selected_validation"]
    }
    if set(receipt["files"]) != expected:
        raise ValueError("Common feature population changed")
    for path, digest in receipt["files"].items():
        if original.protocol.sha(path) != digest:
            raise ValueError(f"Shared cache changed: {path}")
    return spec, cohort


def initial_digest(model, common_only=False):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if common_only and name.startswith("projection."):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(arm):
    spec, cohort = verify_cache()
    marker = OUT / f"{arm}_STARTED.json"
    if marker.exists():
        raise RuntimeError("Corrected arm already started; automatic retry/resume prohibited")
    device = spec["frontends"][arm]["device"]
    if not torch.xpu.is_available():
        raise RuntimeError("Required XPU unavailable")
    torch.xpu.set_device(device)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    import torchaudio

    original.protocol.freeze(
        marker,
        {
            "corrected_protocol_sha256": original.protocol.sha(OUT / "preregistration.json"),
            "original_protocol_sha256": original.protocol.sha(
                original.OUT / "preregistration.json"
            ),
            "cache_integrity_sha256": original.protocol.sha(original.OUT / "cache_integrity.json"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "numpy": np.__version__,
            "device": device,
            "device_properties": str(torch.xpu.get_device_properties(device)),
            "driver_runtime": str(torch.__config__.show()),
            "deterministic_algorithms": True,
            "precision": "FP32",
            "cpu_threads": 2,
            "closed_original_P10_resumed": False,
        },
    )
    scaler = json.loads((original.OUT / f"{arm}_scaler.json").read_text())
    original.protocol.freeze(OUT / f"{arm}_scaler.json", scaler)
    model = ScaleCTC(
        spec["frontends"][arm]["dimensions"],
        width=256,
        layers=12,
        heads=4,
        expansion=1024,
        kernel=15,
        dropout=0.0,
    )
    original.protocol.freeze(
        OUT / f"{arm}_initial_state.json",
        {
            "full_state_sha256": initial_digest(model),
            "common_encoder_head_sha256": initial_digest(model, True),
            "parameters": sum(p.numel() for p in model.parameters()),
            "fresh_seed": 0,
            "engineering_weights_loaded": False,
            "pretrained_weights_loaded": False,
        },
    )
    prior_initial = json.loads(
        Path(f"outputs/rclat_p10_corrected_v1/{arm}_initial_state.json").read_text()
    )
    if initial_digest(model) != prior_initial["full_state_sha256"]:
        raise ValueError("Fresh initial state does not match frozen seed-0 predecessor")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    check_parameters_and_optimizer(model, optimizer)
    training = cohort["selected_training"]
    validation = sorted(cohort["selected_validation"], key=lambda r: r["uid"])
    targets = {r["uid"]: encode_text(r["transcript"]) for r in training}
    total_steps = 30 * math.ceil(len(training) / 4)
    checkpoints = CHECKPOINTS / arm
    checkpoints.mkdir(parents=True, exist_ok=True)
    step = 0
    history = []

    def batch(rows):
        arrays = [frozen_scale(*original.sequence(r["uid"], arm), scaler) for r in rows]
        lengths = torch.tensor([len(x) for x, _ in arrays])
        values = pad_sequence([torch.from_numpy(x) for x, _ in arrays], batch_first=True).to(device)
        masks = pad_sequence([torch.from_numpy(m) for _, m in arrays], batch_first=True).to(device)
        return values, masks, lengths

    with (OUT / f"{arm}_steps.jsonl").open("x", buffering=1) as journal:
        for epoch in range(1, 31):
            model.train()
            start = time.perf_counter()
            ordered = original.epoch_order(training, epoch)
            losses = []
            for pos in range(0, len(ordered), 4):
                rows = ordered[pos : pos + 4]
                step += 1
                context = {
                    "epoch": epoch,
                    "step": step,
                    "epoch_step": pos // 4 + 1,
                    "uids": [r["uid"] for r in rows],
                }
                journal.write(json.dumps({**context, "event": "begin"}) + "\n")
                for group in optimizer.param_groups:
                    group["lr"] = original.learning_rate(step, total_steps)
                optimizer.zero_grad(set_to_none=True)
                inputs = batch(rows)
                journal.write(
                    json.dumps(
                        {
                            **context,
                            "event": "forward_begin",
                            "feature_shape": list(inputs[0].shape),
                            "native_lengths": inputs[2].tolist(),
                        }
                    )
                    + "\n"
                )
                if step == 1 or step % 100 == 0 or 7700 <= step <= 7800:
                    journal.write(
                        json.dumps({**context, "event": "resource_snapshot", **telemetry(device)})
                        + "\n"
                    )
                logits, lengths = model(*inputs)
                del inputs
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite logits")
                loss = ctc_objective(logits, lengths, [targets[r["uid"]] for r in rows])
                loss.backward()
                gradients = check_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm")
                journal.write(
                    json.dumps(
                        {
                            **context,
                            "event": "backward_complete",
                            "loss": float(loss.detach()),
                            "gradient_norm": float(norm),
                            "gradient_tensors_checked": gradients,
                        }
                    )
                    + "\n"
                )
                journal.write(json.dumps({**context, "event": "optimizer_begin"}) + "\n")
                optimizer.step()
                state_checks = check_parameters_and_optimizer(model, optimizer)
                journal.write(
                    json.dumps({**context, "event": "step_complete", **state_checks}) + "\n"
                )
                losses.append(float(loss.detach()))
            torch.xpu.synchronize(device)
            training_seconds = time.perf_counter() - start
            journal.write(
                json.dumps(
                    {"event": "validation_begin", "epoch": epoch, "step": step, **telemetry(device)}
                )
                + "\n"
            )
            model.eval()
            predictions = []
            with torch.no_grad():
                for pos in range(0, len(validation), 4):
                    rows = validation[pos : pos + 4]
                    inputs = batch(rows)
                    journal.write(
                        json.dumps(
                            {
                                "event": "validation_forward_begin",
                                "epoch": epoch,
                                "validation_batch": pos // 4 + 1,
                                "uids": [r["uid"] for r in rows],
                                "feature_shape": list(inputs[0].shape),
                                "native_lengths": inputs[2].tolist(),
                            }
                        )
                        + "\n"
                    )
                    logits, lengths = model(*inputs)
                    del inputs
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Nonfinite validation logits")
                    for row, logit, length in zip(rows, logits.cpu(), lengths):
                        hypothesis = decode_logits(logit, length)
                        predictions.append(
                            {
                                "uid": row["uid"],
                                "speaker": row["speaker"],
                                "hypothesis": hypothesis,
                                **text_errors(row["transcript"], hypothesis),
                            }
                        )
            item = {
                "epoch": epoch,
                "mean_batch_training_loss": float(np.mean(losses)),
                "training_wall_seconds_including_cache_io": training_seconds,
                "epoch_wall_seconds_including_validation": time.perf_counter() - start,
                **original.aggregate(predictions),
            }
            journal.write(
                json.dumps(
                    {"event": "validation_end", "epoch": epoch, "step": step, **telemetry(device)}
                )
                + "\n"
            )
            history.append(item)
            checkpoint = checkpoints / f"epoch_{epoch:02d}.pt"
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
                    "corrected_protocol_sha256": original.protocol.sha(
                        OUT / "preregistration.json"
                    ),
                },
                checkpoint,
            )
            original.protocol.freeze(
                OUT / f"{arm}_epoch_{epoch:02d}.json",
                {
                    **item,
                    "predictions": predictions,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": original.protocol.sha(checkpoint),
                },
            )
            journal.write(
                json.dumps(
                    {"event": "checkpoint_saved", "epoch": epoch, "step": step, **telemetry(device)}
                )
                + "\n"
            )
            print(json.dumps({"arm": arm, **item}), flush=True)
    selected = min(history, key=lambda r: (r["speaker_macro_cer"], r["epoch"]))
    original.protocol.freeze(
        OUT / f"{arm}_result.json",
        {
            "selected": selected,
            "final": history[-1],
            "learning_gate_passed": selected["speaker_macro_cer"] < 0.5
            and selected["nonempty_fraction"] >= 0.95,
            "status": "CORRECTED_P10_DEVELOPMENT_VALIDATION_ONLY",
            "stage2_ready": False,
            "closed_original_P10_resumed": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["verify", "train"])
    parser.add_argument("--arm", choices=["mel", "combined"])
    args = parser.parse_args()
    if args.action == "verify":
        _, cohort = verify_cache()
        print(
            f"Corrected protocol and5678shared cache entries verified;no training run. Training n={len(cohort['selected_training'])}"
        )
    elif args.arm is None:
        parser.error("train requires arm")
    else:
        try:
            train(args.arm)
        except Exception:
            failure = OUT / f"{args.arm}_FAILURE.json"
            if not failure.exists():
                last = None
                journal = OUT / f"{args.arm}_steps.jsonl"
                if journal.exists():
                    lines = journal.read_text().splitlines()
                    if lines:
                        last = json.loads(lines[-1])
                original.protocol.freeze(
                    failure,
                    {
                        "traceback": traceback.format_exc(),
                        "last_journal_event": last,
                        "automatic_retry_permitted": False,
                    },
                )
            raise
