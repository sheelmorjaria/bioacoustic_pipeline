"""Executable P10 policy. Separate cache and train actions; never starts on import.

No final-test path exists. A started arm cannot be silently retried.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from threadpoolctl import threadpool_limits

from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.diagnostics.rclat import text_errors
from bioacoustic_pipeline.representation.rclat_ctc import (
    acoustic_features,
    ctc_objective,
    decode_logits,
    encode_text,
    fit_scaler,
)
from bioacoustic_pipeline.representation.rclat_scale_v3 import ScaleCTC
import rclat_p10_protocol as protocol

OUT = protocol.OUT
CACHE = Path("cache/rclat_p10")


def check():
    spec = json.loads((OUT / "preregistration.json").read_text())
    for path, digest in spec["dependencies"].items():
        if protocol.sha(path) != digest:
            raise ValueError(f"Frozen dependency changed: {path}")
    if protocol.software_versions() != spec["software"]:
        raise ValueError("Frozen scientific software versions changed")
    return spec, json.loads((OUT / "cohort.json").read_text())


def cache():
    _, cohort = check()
    receipt = OUT / "cache_integrity.json"
    if receipt.exists():
        raise RuntimeError("Cache already frozen")
    CACHE.mkdir(parents=True, exist_ok=True)
    hashes = {}
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        for row in cohort["selected_training"] + cohort["selected_validation"]:
            if protocol.sha(row["path"]) != row["source_sha256"]:
                raise ValueError("Source changed")
            audio, rate = sf.read(row["path"])
            features = acoustic_features(audio, rate)
            if len(features["mel"]) != row["feature_frames"]:
                raise ValueError("Feature alignment changed")
            target = CACHE / f"{row['uid']}.npz"
            np.savez_compressed(target, **features)
            hashes[str(target)] = protocol.sha(target)
    protocol.freeze(
        receipt,
        {
            "files": hashes,
            "wall_seconds": time.perf_counter() - start,
            "preregistration_sha256": protocol.sha(OUT / "preregistration.json"),
        },
    )


def sequence(uid, arm):
    with np.load(CACHE / f"{uid}.npz") as z:
        if arm == "mel":
            return z["mel"], z["mel_valid"]
        return np.concatenate([z["spectral"], z["physical"]], 1), np.concatenate(
            [z["spectral_valid"], z["physical_valid"]], 1
        )


def epoch_order(rows, epoch):
    return sorted(
        rows,
        key=lambda r: hashlib.sha256(
            f"{protocol.SALT}:epoch:{epoch}:{r['uid']}".encode()
        ).hexdigest(),
    )


def learning_rate(step, steps):
    if step <= 500:
        return 1e-6 + (1e-3 - 1e-6) * step / 500
    fraction = (step - 500) / max(1, steps - 500)
    return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * fraction))


def aggregate(predictions):
    speakers = defaultdict(list)
    for row in predictions:
        speakers[row["speaker"]].append(row)
    by_speaker = []
    for speaker, rows in sorted(speakers.items()):
        chars = sum(r["reference_characters"] for r in rows)
        words = sum(r["reference_words"] for r in rows)
        by_speaker.append(
            {
                "speaker": speaker,
                "cer": sum(r["character_errors"] for r in rows) / chars,
                "wer": sum(r["word_errors"] for r in rows) / words,
            }
        )
    return {
        "speaker_macro_cer": float(np.mean([r["cer"] for r in by_speaker])),
        "speaker_macro_wer": float(np.mean([r["wer"] for r in by_speaker])),
        "pooled_cer": sum(r["character_errors"] for r in predictions)
        / sum(r["reference_characters"] for r in predictions),
        "pooled_wer": sum(r["word_errors"] for r in predictions)
        / sum(r["reference_words"] for r in predictions),
        "nonempty_fraction": sum(bool(r["hypothesis"].strip()) for r in predictions)
        / len(predictions),
        "speaker_results": by_speaker,
    }


def train(arm):
    spec, cohort = check()
    receipt = json.loads((OUT / "cache_integrity.json").read_text())
    if receipt["preregistration_sha256"] != protocol.sha(OUT / "preregistration.json"):
        raise ValueError("Cache protocol mismatch")
    for path, digest in receipt["files"].items():
        if protocol.sha(path) != digest:
            raise ValueError("Cache changed")
    marker = OUT / f"{arm}_STARTED.json"
    if marker.exists():
        raise RuntimeError("Arm already started; no automatic retry/resume implemented")
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

    protocol.freeze(
        marker,
        {
            "preregistration_sha256": protocol.sha(OUT / "preregistration.json"),
            "cache_integrity_sha256": protocol.sha(OUT / "cache_integrity.json"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "numpy": np.__version__,
            "device": device,
            "device_properties": str(torch.xpu.get_device_properties(device)),
            "deterministic_algorithms": True,
            "precision": "FP32",
            "cpu_threads": 2,
            "driver_runtime": str(torch.__config__.show()),
        },
    )
    training = cohort["selected_training"]
    validation = sorted(cohort["selected_validation"], key=lambda r: r["uid"])
    scaler = fit_scaler(sequence(r["uid"], arm) for r in training)
    protocol.freeze(OUT / f"{arm}_scaler.json", scaler)
    model = ScaleCTC(
        spec["frontends"][arm]["dimensions"],
        width=256,
        layers=12,
        heads=4,
        expansion=1024,
        kernel=15,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    targets = {r["uid"]: encode_text(r["transcript"]) for r in training}
    steps = 30 * math.ceil(len(training) / 4)
    step = 0
    history = []
    checkpoint_dir = Path("checkpoints/rclat_p10") / arm
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def batch(rows):
        arrays = [frozen_scale(*sequence(r["uid"], arm), scaler) for r in rows]
        lengths = torch.tensor([len(a[0]) for a in arrays])
        values = pad_sequence([torch.from_numpy(a[0]) for a in arrays], batch_first=True).to(device)
        masks = pad_sequence([torch.from_numpy(a[1]) for a in arrays], batch_first=True).to(device)
        return values, masks, lengths

    for epoch in range(1, 31):
        started = time.perf_counter()
        model.train()
        ordered = epoch_order(training, epoch)
        losses = []
        for pos in range(0, len(ordered), 4):
            rows = ordered[pos : pos + 4]
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, steps)
            optimizer.zero_grad(set_to_none=True)
            logits, lengths = model(*batch(rows))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite neural logits")
            loss = ctc_objective(logits, lengths, [targets[r["uid"]] for r in rows])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite parameter gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.xpu.synchronize(device)
        training_seconds = time.perf_counter() - started
        model.eval()
        predictions = []
        with torch.no_grad():
            for pos in range(0, len(validation), 4):
                rows = validation[pos : pos + 4]
                logits, lengths = model(*batch(rows))
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite validation logits")
                logits = logits.cpu()
                for row, logit, length in zip(rows, logits, lengths):
                    hypothesis = decode_logits(logit, length)
                    predictions.append(
                        {
                            "uid": row["uid"],
                            "speaker": row["speaker"],
                            "hypothesis": hypothesis,
                            **text_errors(row["transcript"], hypothesis),
                        }
                    )
        metrics = aggregate(predictions)
        item = {
            "epoch": epoch,
            "mean_batch_training_loss": float(np.mean(losses)),
            "training_wall_seconds_including_cache_io": training_seconds,
            "epoch_wall_seconds_including_validation": time.perf_counter() - started,
            **metrics,
        }
        history.append(item)
        protocol.freeze(OUT / f"{arm}_epoch_{epoch:02d}.json", {**item, "predictions": predictions})
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
                "torch_rng": torch.get_rng_state(),
                "xpu_rng": torch.xpu.get_rng_state_all(),
                "python_rng": random.getstate(),
                "preregistration_sha256": protocol.sha(OUT / "preregistration.json"),
            },
            checkpoint_dir / f"epoch_{epoch:02d}.pt",
        )
        print(json.dumps({"arm": arm, **item}), flush=True)
    selected = min(history, key=lambda r: (r["speaker_macro_cer"], r["epoch"]))
    protocol.freeze(
        OUT / f"{arm}_result.json",
        {
            "selected": selected,
            "final": history[-1],
            "learning_gate_passed": selected["speaker_macro_cer"] < 0.5
            and selected["nonempty_fraction"] >= 0.95,
            "status": "DEVELOPMENT_VALIDATION_ONLY",
            "stage2_ready": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["cache", "train"])
    parser.add_argument("--arm", choices=["mel", "combined"])
    args = parser.parse_args()
    if args.action == "cache":
        cache()
    elif args.arm is None:
        parser.error("train requires --arm")
    else:
        try:
            train(args.arm)
        except Exception:
            failure = OUT / f"{args.arm}_FAILURE.json"
            if not failure.exists():
                protocol.freeze(
                    failure, {"error": traceback.format_exc(), "automatic_retry_permitted": False}
                )
            raise
