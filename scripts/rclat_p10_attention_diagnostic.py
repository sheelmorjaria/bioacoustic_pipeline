"""No-update operator-level attention VJP comparison on isolated training state."""

import argparse
import json
import math

import torch
from torch import nn

import rclat_p10_numerical_diagnostic as diagnosis
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text

OUT = diagnosis.OUT


def prepare():
    diagnosis.check()
    paths = [__file__] + [str(OUT / f"{arm}_isolated_state.pt") for arm in ["mel", "combined"]]
    diagnosis.original.protocol.freeze(
        OUT / "attention_operator_plan.json",
        {
            "scope": "No-update training-only follow-up. Capture final-block attention input and upstream VJP, then compare identical inputs on CPU/XPU.",
            "variants": [
                "native",
                "contiguous_softmax_gradient",
                "composite_softmax",
                "contiguous_qkv",
            ],
            "comparisons": "Output and intermediate VJP values versus CPU reference; exact shared input and upstream gradient. No validation or scientific fitting.",
            "dependencies": {str(p): diagnosis.original.protocol.sha(p) for p in paths},
            "optimizer_updates": 0,
            "closed_p10_resumed": False,
        },
    )


def attention(q, weight, bias, valid, variant):
    batch, time, width = q.shape
    heads = 4
    dim = width // heads
    packed = nn.functional.linear(q, weight, bias)
    query, key, value = [
        p.reshape(batch, time, heads, dim).transpose(1, 2) for p in packed.chunk(3, dim=-1)
    ]
    if variant == "contiguous_qkv":
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    scaled = query / math.sqrt(dim)
    scores = torch.matmul(scaled, key.transpose(-2, -1))
    masked = scores.masked_fill(~valid[:, None, None, :], -10000.0)
    if variant == "composite_softmax":
        numerator = (masked - masked.amax(-1, keepdim=True)).exp()
        probabilities = numerator / numerator.sum(-1, keepdim=True)
    else:
        probabilities = nn.functional.softmax(masked, dim=-1)
    if variant == "contiguous_softmax_gradient":
        probabilities.register_hook(lambda grad: grad.contiguous())
    context = torch.matmul(probabilities, value)
    attended = context.transpose(1, 2).reshape(batch, time, width)
    tensors = {
        "q": q,
        "packed": packed,
        "scaled_query": scaled,
        "key": key,
        "value": value,
        "scores": scores,
        "masked_scores": masked,
        "probabilities": probabilities,
        "context": context,
        "attended": attended,
    }
    for t in tensors.values():
        t.retain_grad()
    return attended, tensors


def run(arm):
    spec, cohort, _ = diagnosis.check()
    plan = json.loads((OUT / "attention_operator_plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if diagnosis.original.protocol.sha(path) != digest:
            raise ValueError("Attention probe changed")
    device = spec["frontends"][arm]["device"]
    diagnosis.seed(device)
    saved = torch.load(OUT / f"{arm}_isolated_state.pt", map_location="cpu", weights_only=True)
    model = diagnosis.model_for(arm, device)
    model.load_state_dict(saved["model"])
    mapping = {r["uid"]: r for r in cohort["selected_training"]}
    rows = [mapping[uid] for uid in saved["uids"]]
    inputs = diagnosis.batch_for(rows, arm, device)
    capture = {}

    def q_hook(module, args, output):
        capture["q"] = output.detach().cpu()

    def out_hook(module, args):
        capture["attended"] = args[0].detach().cpu()

        def backward(gradient):
            capture["upstream"] = gradient.detach().cpu()

        args[0].register_hook(backward)

    model.blocks[-1].attention_norm.register_forward_hook(q_hook)
    model.blocks[-1].attention.out_proj.register_forward_pre_hook(out_hook)
    logits, lengths = model(*inputs)
    loss = ctc_objective(logits, lengths, [encode_text(r["transcript"]) for r in rows])
    loss.backward()
    capture["valid"] = torch.arange(logits.shape[1])[None] < lengths[:, None]
    capture["weight"] = model.blocks[-1].attention.in_proj_weight.detach().cpu()
    capture["bias"] = model.blocks[-1].attention.in_proj_bias.detach().cpu()
    torch.save(capture, OUT / f"{arm}_attention_operator_inputs.pt")
    del model, logits, loss
    torch.xpu.empty_cache()
    reference = None
    reports = []
    for target, variant in [("cpu", "native")] + [(device, v) for v in plan["variants"]]:
        q = capture["q"].to(target).clone().requires_grad_()
        w = capture["weight"].to(target).clone().requires_grad_()
        b = capture["bias"].to(target).clone().requires_grad_()
        output, tensors = attention(q, w, b, capture["valid"].to(target), variant)
        output.backward(capture["upstream"].to(target))
        observed = {
            name: {"value": t.detach().cpu(), "gradient": t.grad.detach().cpu()}
            for name, t in tensors.items()
        }
        observed["weight"] = {"value": w.detach().cpu(), "gradient": w.grad.detach().cpu()}
        observed["bias"] = {"value": b.detach().cpu(), "gradient": b.grad.detach().cpu()}
        if reference is None:
            reference = observed
        comparisons = {}
        for name, values in observed.items():
            comparisons[name] = {}
            for kind, tensor in values.items():
                expected = reference[name][kind]
                diff = tensor.double() - expected.double()
                comparisons[name][kind] = {
                    **diagnosis.tensor_stats(tensor),
                    "max_abs_difference_from_cpu": diagnosis.scalar(diff.abs().max()),
                    "relative_l2_difference_from_cpu": diagnosis.scalar(
                        torch.linalg.vector_norm(diff)
                        / torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
                    ),
                }
        reports.append({"device": target, "variant": variant, "comparisons": comparisons})
        print(
            json.dumps(
                {
                    "device": target,
                    "variant": variant,
                    "q_gradient": comparisons["q"]["gradient"],
                    "softmax_gradient": comparisons["masked_scores"]["gradient"],
                }
            ),
            flush=True,
        )
    diagnosis.original.protocol.freeze(
        OUT / f"{arm}_attention_operator_results.json",
        {
            "arm": arm,
            "training_uids": saved["uids"],
            "loss_for_capture": "Original FP32 CPU CTC",
            "upstream": diagnosis.tensor_stats(capture["upstream"]),
            "reports": reports,
            "optimizer_updates": 0,
            "validation_used": False,
            "closed_p10_resumed": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--arm", choices=["mel", "combined"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.arm is None:
        parser.error("arm required")
    else:
        run(args.arm)
