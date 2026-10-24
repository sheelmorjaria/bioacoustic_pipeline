"""Training-only closed-checkpoint diagnostics; never update a closed model."""

import json
from pathlib import Path

import torch

import rclat_study_b as closed
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.representation.rclat_ctc import (
    CausalCTC,
    ctc_objective,
    decode_logits,
    encode_text,
)


def main():
    output = Path("outputs/rclat_scale_training_diagnostics.json")
    if output.exists():
        raise RuntimeError("Diagnostic already exists")
    closed.check()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    records = [r for r in json.loads(closed.AUDIT.read_text())["records"] if r["split"] == "train"]
    cache = {r["uid"]: Path(r["path"]) for r in json.loads(closed.FEATURES.read_text())["records"]}
    results = {}
    for arm in closed.ARMS:
        path = Path(f"checkpoints/rclat_study_b_{arm}.pt")
        before = closed.prior.sha(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = CausalCTC(saved["model_arm"], saved["input_dim"], saved["width"]).eval()
        model.load_state_dict(saved["state_dict"])
        total_blank = 0.0
        frames = 0
        empty = 0
        outputs = []
        first = None
        for source in records:
            x, mask = closed.sequence(cache[source["uid"]], arm)
            x, mask = frozen_scale(x, mask, saved["scaler"])
            values = torch.from_numpy(x)[None]
            valid = torch.from_numpy(mask)[None]
            with torch.no_grad():
                logits, _ = model(values, valid)
                prob = logits.softmax(-1)[0, :, 0]
                total_blank += float(prob.sum())
                frames += len(x)
                decoded = decode_logits(logits[0], len(x))
                empty += not decoded
                outputs.append(
                    {
                        "uid": source["uid"],
                        "predicted_characters": len(decoded.replace(" ", "")),
                        "reference_characters": len(source["transcript"].replace(" ", "")),
                        "input_frames": len(x),
                        "ctc_minimum": len(encode_text(source["transcript"]))
                        + int(
                            (
                                encode_text(source["transcript"])[1:]
                                == encode_text(source["transcript"])[:-1]
                            ).sum()
                        ),
                    }
                )
            if first is None:
                first = (values, valid, source)
        values, valid, source = first
        logits, _ = model(values, valid)
        loss = ctc_objective(
            logits, torch.tensor([values.shape[1]]), [encode_text(source["transcript"])]
        )
        loss.backward()
        norm = float(
            torch.sqrt(
                sum(
                    (
                        p.grad.double().square().sum()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )
            )
        )
        if not norm > 0 or not all(
            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
        ):
            raise ValueError("Closed checkpoint lacks finite connected gradients")
        if closed.prior.sha(path) != before:
            raise ValueError("Closed checkpoint mutated")
        results[arm] = {
            "training_utterances": len(records),
            "frame_weighted_blank_probability": total_blank / frames,
            "empty_training_transcripts": empty,
            "first_training_ctc_loss": float(loss.detach()),
            "read_only_gradient_l2": norm,
            "checkpoint_unchanged": True,
            "records": outputs,
        }
    closed.check()
    closed.prior.freeze_json(
        output,
        {
            "scope": "Read-only training-only diagnosis of closed Study B checkpoints; no validation or evaluation scoring and no optimizer steps.",
            "arms": results,
            "causal_explanation_established": False,
            "new_scientific_training_performed": False,
            "stage2_ready": False,
        },
    )
    print(
        json.dumps(
            {
                a: {
                    k: r[k]
                    for k in [
                        "frame_weighted_blank_probability",
                        "empty_training_transcripts",
                        "read_only_gradient_l2",
                    ]
                }
                for a, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
