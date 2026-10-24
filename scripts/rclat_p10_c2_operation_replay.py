"""Bounded synthetic Conformer convolution-path replay including validation shapes."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from rclat_p10_c1_shape_replay import telemetry
from bioacoustic_pipeline.representation.rclat_scale_v4 import Block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()
    out = Path("outputs/rclat_p10_c2_operation_replay")
    if (out / "plan.json").exists():
        raise RuntimeError("Existing diagnostic; no automatic retry")
    out.mkdir(exist_ok=True)
    journal = Path("outputs/rclat_p10_corrected_v2/mel_steps.jsonl")
    events = [json.loads(s) for s in journal.read_text().splitlines()]
    shapes = [r for r in events if r["event"] in ("forward_begin", "validation_forward_begin")]
    plan = dict(
        source_journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        maximum_seconds=args.max_seconds,
        events=len(shapes),
        device="xpu:0",
        layers=12,
        scientific_training=False,
        optimizer_updates=0,
        synthetic_inputs=True,
        preserved="Recorded training/validation shape order; convolution norm,point-in,GLU,depth,frame-norm,SiLU,point-out,residual path in12blocks;train/eval and grad/no_grad transitions.",
        omitted="Attention,feedforward,subsampling,projection,CTC,optimizer,checkpoint serialization,real activations,original runtime state.",
        stop="First exception/nonfinite output or time bound; do not retry",
        target_test="Fresh-process target shape before sequence; final target if historical sequence completes",
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = "xpu:0"
    torch.xpu.set_device(device)
    blocks = nn.ModuleList([Block(256, 4, 1024, 15, 0.0) for _ in range(12)]).to(device)

    def forward(x):
        for block in blocks:
            y = block.point_in(block.convolution_norm(x).transpose(1, 2))
            y = nn.functional.glu(y, dim=1)
            y = block.depth(y).transpose(1, 2)
            y = nn.functional.silu(block.frame_norm(y))
            y = block.point_out(y.transpose(1, 2)).transpose(1, 2)
            x = x + y
        return x

    def run(batch, frames, training):
        blocks.train(training)
        blocks.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            x = torch.randn(batch, frames, 256, device=device, requires_grad=training)
            y = forward(x)
            if training:
                y.square().mean().backward()
            if not bool(torch.isfinite(y).all()):
                raise FloatingPointError("Nonfinite output")
            if training and not bool(torch.isfinite(x.grad).all()):
                raise FloatingPointError("Nonfinite input gradient")
        del x, y

    start = time.monotonic()
    record = dict(completed_events=0, snapshots=[])
    with (out / "progress.jsonl").open("x", buffering=1) as stream:
        try:
            run(4, 417, True)
            record["fresh_target_passed"] = True
            for i, event in enumerate(shapes, 1):
                if time.monotonic() - start >= args.max_seconds:
                    record["stop_reason"] = "wall_time_bound"
                    break
                batch, native, _ = event["feature_shape"]
                frames = (native + 3) // 4
                training = event["event"] == "forward_begin"
                record["active_event"] = dict(
                    index=i,
                    epoch=event["epoch"],
                    step=event.get("step"),
                    training=training,
                    batch=batch,
                    frames=frames,
                    uids=event.get("uids"),
                )
                run(batch, frames, training)
                record["completed_events"] = i
                if i == 1 or i % 100 == 0 or i == len(shapes):
                    snap = dict(
                        **record["active_event"],
                        seconds=time.monotonic() - start,
                        **telemetry(device),
                    )
                    record["snapshots"].append(snap)
                    stream.write(json.dumps(snap) + "\n")
                    print(json.dumps(snap), flush=True)
            else:
                run(4, 417, True)
                record["final_target_passed"] = True
                record["stop_reason"] = "completed_recorded_operation_sequence"
        except Exception as exc:
            record.update(
                stop_reason="exception", error=repr(exc), failure_telemetry=telemetry(device)
            )
        record["seconds"] = time.monotonic() - start
        record["historical_primitive_failure_reproduced"] = (
            "could not create a primitive" in record.get("error", "")
        )
        (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "snapshots"}), flush=True)


if __name__ == "__main__":
    main()
