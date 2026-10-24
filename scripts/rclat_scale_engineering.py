"""Bounded, training-only XPU throughput and memorisation test; not an ASR study."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import rclat_study_b as closed
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.diagnostics.rclat import text_errors
from bioacoustic_pipeline.representation.rclat_ctc import (
    ctc_objective,
    decode_logits,
    encode_text,
    fit_scaler,
)
from bioacoustic_pipeline.representation.rclat_scale import ScaleCTC
from bioacoustic_pipeline.utils.devices import resolve_device

PLAN = Path("outputs/rclat_scale_engineering_plan.json")
KINDS = {"mel": ("B0", 64, "xpu:0"), "combined": ("B2", 76, "xpu:1")}


def sources():
    records = [r for r in json.loads(closed.AUDIT.read_text())["records"] if r["split"] == "train"]
    cache = {r["uid"]: Path(r["path"]) for r in json.loads(closed.FEATURES.read_text())["records"]}
    return records, cache


def prepare():
    closed.check()
    rows, _ = sources()
    ordered = sorted(rows, key=lambda r: (r["duration_seconds"], r["uid"]))
    invalid = []
    for r in rows:
        frames = 1 + (r["samples"] - 640) // 160
        tokens = encode_text(r["transcript"])
        minimum = len(tokens) + int((tokens[1:] == tokens[:-1]).sum())
        if (frames + 3) // 4 < minimum:
            invalid.append(r["uid"])
    if invalid:
        raise ValueError("Subsampling makes training CTC infeasible; no adaptive exclusions")
    paths = [
        Path(__file__),
        Path("src/bioacoustic_pipeline/representation/rclat_scale.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_ctc.py"),
        Path("tests/test_rclat_scale.py"),
        closed.AUDIT,
        closed.FEATURES,
        Path("outputs/rclat_scale_training_diagnostics.json"),
    ]
    closed.prior.freeze_json(
        PLAN,
        {
            "scope": "Engineering only: known Study B training sources, no protected/validation scoring.",
            "training_pool_utterances": len(rows),
            "subsampling_ctc_infeasible": invalid,
            "throughput_uids": [r["uid"] for r in ordered[-4:]],
            "memorisation_uids": [r["uid"] for r in ordered[:4]],
            "architecture": "Offline Conformer-style4blocks,width128,4heads,FF512,depthwise kernel15,framewiseLayerNorm,4xmasked temporal subsampling. FP32,mathSDPA,dropout0.",
            "causal_or_streaming_claim": False,
            "policy": "AdamW lr0.001 weight_decay0.01 clip5;seed0;2warmup+5timed updates on4long training clips,then300fixed updates on4short training clips. No stopping/tuning from outcomes.",
            "memorisation_gate": "All4decoded training examples nonempty and pooled training CER<=0.10;not a heldout learning gate.",
            "throughput_scope": "Precomputed feature tensors from RAM; CPU/XPU transfer,CPU CTC,backward and optimizer timed. Excludes waveform feature extraction,storage I/O,larger batch tuning and distributed training.",
            "devices": {a: v[2] for a, v in KINDS.items()},
            "dependencies": {str(p): closed.prior.sha(p) for p in paths},
            "scientific_scaling_preregistered": False,
            "stage2_ready": False,
        },
    )
    print("Engineering policy fixed; no scientific scaling experiment registered.")


def check():
    plan = json.loads(PLAN.read_text())
    for p, digest in plan["dependencies"].items():
        if closed.prior.sha(Path(p)) != digest:
            raise ValueError("Engineering dependency changed")
    closed.check()
    return plan


def run(arm):
    plan = check()
    device = resolve_device(KINDS[arm][2])
    output = Path(f"outputs/rclat_scale_engineering_{arm}.json")
    marker = Path(f"outputs/rclat_scale_engineering_{arm}_STARTED.json")
    if output.exists() or marker.exists():
        raise RuntimeError("Engineering arm already started")
    closed.prior.freeze_json(
        marker, {"engineering_plan_sha256": closed.prior.sha(PLAN), "device": device}
    )
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    rows, cache = sources()
    mapping = {r["uid"]: r for r in rows}
    kind, dimension, _ = KINDS[arm]
    arrays = {r["uid"]: closed.sequence(cache[r["uid"]], kind) for r in rows}
    scaler = fit_scaler(arrays.values())
    data = {uid: frozen_scale(x, mask, scaler) for uid, (x, mask) in arrays.items()}
    model = ScaleCTC(dimension).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    targets = {r["uid"]: encode_text(r["transcript"]) for r in rows}
    traces = []
    timings = []

    def step(uids):
        start = time.perf_counter()
        lengths = torch.tensor([len(data[uid][0]) for uid in uids])
        values = pad_sequence(
            [torch.from_numpy(data[uid][0]) for uid in uids], batch_first=True
        ).to(device)
        masks = pad_sequence([torch.from_numpy(data[uid][1]) for uid in uids], batch_first=True).to(
            device
        )
        torch.xpu.synchronize()
        a = time.perf_counter()
        optimizer.zero_grad()
        logits, out_lengths = model(values, masks, lengths)
        torch.xpu.synchronize()
        b = time.perf_counter()
        loss = ctc_objective(logits, out_lengths, [targets[uid] for uid in uids])
        c = time.perf_counter()
        loss.backward()
        torch.xpu.synchronize()
        d = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        torch.xpu.synchronize()
        end = time.perf_counter()
        return float(loss.detach()), {
            "total_seconds": end - start,
            "input_transfer_seconds": a - start,
            "forward_seconds": b - a,
            "cpu_ctc_forward_seconds": c - b,
            "backward_including_cpu_ctc_seconds": d - c,
            "optimizer_and_clip_seconds": end - d,
        }

    for _ in range(2):
        step(plan["throughput_uids"])
    torch.xpu.reset_peak_memory_stats(device)
    for _ in range(5):
        loss, timing = step(plan["throughput_uids"])
        timings.append(timing)
    benchmark_peak = torch.xpu.max_memory_allocated(device)
    uids = plan["memorisation_uids"]
    for i in range(1, 301):
        loss, _ = step(uids)
        traces.append(loss)
        if i % 50 == 0:
            print(arm, "training-only memorisation step", i, "loss", loss, flush=True)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for uid in uids:
            x, mask = data[uid]
            logits, lengths = model(
                torch.from_numpy(x)[None].to(device),
                torch.from_numpy(mask)[None].to(device),
                torch.tensor([len(x)]),
            )
            text = decode_logits(logits[0], int(lengths[0]))
            metrics = text_errors(mapping[uid]["transcript"], text)
            predictions.append(
                {
                    "uid": uid,
                    "reference": mapping[uid]["transcript"],
                    "prediction": text,
                    "metrics": metrics,
                    "blank_probability": float(
                        logits[0, : int(lengths[0])].softmax(-1)[:, 0].mean().cpu()
                    ),
                }
            )
    seconds = sum(mapping[uid]["duration_seconds"] for uid in plan["throughput_uids"])
    mean = {k: float(np.mean([r[k] for r in timings])) for k in timings[0]}
    cer = sum(r["metrics"]["character_errors"] for r in predictions) / sum(
        r["metrics"]["reference_characters"] for r in predictions
    )
    nonempty = sum(bool(r["prediction"]) for r in predictions)
    check()
    closed.prior.freeze_json(
        output,
        {
            "engineering_plan_sha256": closed.prior.sha(PLAN),
            "arm": arm,
            "device": device,
            "device_name": torch.xpu.get_device_name(device),
            "torch": torch.__version__,
            "strict_deterministic_algorithms": True,
            "precision": "FP32",
            "cpu_ctc_loss": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "benchmark_mean": mean,
            "throughput_audio_seconds_per_wall_second": seconds / mean["total_seconds"],
            "benchmark_peak_allocated_bytes": benchmark_peak,
            "training_loss_trace": traces,
            "memorisation_examples": predictions,
            "memorisation_pooled_cer": cer,
            "nonempty_memorisation_outputs": nonempty,
            "memorisation_gate_passed": nonempty == 4 and cer <= 0.10,
            "scientific_validation_or_test_scoring_performed": False,
            "stage2_ready": False,
        },
    )
    print(
        arm,
        "engineering complete; memorisationCER",
        cer,
        "audioseconds/wallsecond",
        seconds / mean["total_seconds"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--arm", choices=list(KINDS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.action == "prepare":
        prepare()
    else:
        if not args.arm:
            parser.error("--arm is required")
        run(args.arm)
