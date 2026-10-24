"""Investigate a failed engineering warm-up using training inputs only; no updates."""

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

import rclat_scale_engineering as engineering
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text, fit_scaler
from bioacoustic_pipeline.representation.rclat_scale_v2 import ScaleCTC


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    plan = engineering.check()
    rows, cache = engineering.sources()
    values = {r["uid"]: engineering.closed.sequence(cache[r["uid"]], "B0") for r in rows}
    scaler = fit_scaler(values.values())
    values = {k: frozen_scale(x, m, scaler) for k, (x, m) in values.items()}
    uids = plan["throughput_uids"]
    mapping = {r["uid"]: r for r in rows}
    x = pad_sequence([torch.from_numpy(values[uid][0]) for uid in uids], batch_first=True).to(
        "xpu:0"
    )
    mask = pad_sequence([torch.from_numpy(values[uid][1]) for uid in uids], batch_first=True).to(
        "xpu:0"
    )
    lengths = torch.tensor([len(values[uid][0]) for uid in uids])
    targets = [encode_text(mapping[uid]["transcript"]) for uid in uids]
    model = ScaleCTC(64).to("xpu:0")
    logits, out_lengths = model(x, mask, lengths)
    comparisons = []
    for dtype in [torch.float32, torch.float64]:
        copied = logits.detach().cpu().to(dtype).requires_grad_()
        loss = ctc_objective(copied, out_lengths, targets)
        (gradient,) = torch.autograd.grad(loss, copied)
        comparisons.append(
            {
                "dtype": str(dtype),
                "loss": float(loss.detach()),
                "loss_finite": bool(torch.isfinite(loss)),
                "logit_gradients_finite": bool(torch.isfinite(gradient).all()),
                "nonfinite_gradient_values": int((~torch.isfinite(gradient)).sum()),
            }
        )
    loss = ctc_objective(logits, out_lengths, targets)
    loss.backward()
    parameters = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    if parameters:
        raise ValueError("Finite mask did not repair parameter gradients")
    path = Path("outputs/rclat_scale_numerical_mask_repair.json")
    engineering.closed.prior.freeze_json(
        path,
        {
            "scope": "Training-only read-only numerical diagnosis;no optimizer steps.",
            "comparisons": comparisons,
            "nonfinite_parameter_gradients": parameters,
            "ctc_input_lengths": out_lengths.tolist(),
            "minimum_target_lengths": [len(t) + int((t[1:] == t[:-1]).sum()) for t in targets],
            "logits_finite": bool(torch.isfinite(logits).all()),
            "scientific_training_performed": False,
            "stage2_ready": False,
        },
    )
    print(
        json.dumps(
            {"comparisons": comparisons, "nonfinite_parameter_gradients": parameters}, indent=2
        )
    )


if __name__ == "__main__":
    main()
