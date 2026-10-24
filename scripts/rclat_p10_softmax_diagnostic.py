"""No-update full-model comparison of a diagnostic composite-softmax candidate.

Only isolated model copies are patched. Frozen production modules are untouched.
"""

import argparse
import json
import types

import torch
from torch import nn

import rclat_p10_numerical_diagnostic as diagnosis
from rclat_p10_attention_diagnostic import attention
from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text

OUT = diagnosis.OUT


def composite_block_forward(self, x, valid):
    """Original Block.forward with only the attention probability path replaced."""
    mask = valid[..., None]
    x = (x + 0.5 * self.ff1(x)).masked_fill(~mask, 0.0)
    q = self.attention_norm(x)
    attended, _ = attention(
        q, self.attention.in_proj_weight, self.attention.in_proj_bias, valid, "composite_softmax"
    )
    attended = self.attention.out_proj(attended)
    x = (x + self.drop(attended)).masked_fill(~mask, 0.0)
    y = self.point_in(self.convolution_norm(x).transpose(1, 2))
    y = nn.functional.glu(y, dim=1).masked_fill(~valid[:, None], 0.0)
    y = self.depth(y).transpose(1, 2)
    y = nn.functional.silu(self.frame_norm(y)).masked_fill(~mask, 0.0)
    y = self.point_out(y.transpose(1, 2)).transpose(1, 2)
    x = (x + self.drop(y)).masked_fill(~mask, 0.0)
    return self.final(x + 0.5 * self.ff2(x)).masked_fill(~mask, 0.0)


def prepare():
    diagnosis.check()
    paths = [__file__, "scripts/rclat_p10_attention_diagnostic.py"]
    diagnosis.original.protocol.freeze(
        OUT / "composite_softmax_plan.json",
        {
            "scope": "No-update full-model numerical candidate; no production-code change or training continuation.",
            "change": "All12block attention probabilities use equivalent stable max/exp/sum expression. Original weights, features, masks, CTC and FP32 retained.",
            "cases": "Isolated third-step state and failing training batch for each arm. CPU native full-CTC reference; XPU candidate fixed CPU upstream VJP; XPU candidate full CPU CTC.",
            "dependencies": {str(p): diagnosis.original.protocol.sha(p) for p in paths},
            "optimizer_updates": 0,
            "validation_used": False,
        },
    )


def run(arm):
    spec, cohort, _ = diagnosis.check()
    plan = json.loads((OUT / "composite_softmax_plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if diagnosis.original.protocol.sha(path) != digest:
            raise ValueError("Candidate changed")
    saved = torch.load(OUT / f"{arm}_isolated_state.pt", map_location="cpu", weights_only=True)
    mapping = {r["uid"]: r for r in cohort["selected_training"]}
    rows = [mapping[uid] for uid in saved["uids"]]
    targets = [encode_text(r["transcript"]) for r in rows]
    device = spec["frontends"][arm]["device"]
    reference_grads = None
    reference_logits = None
    reference_upstream = None
    reports = []
    for target, variant, backward in [
        ("cpu", "native", "ctc"),
        (device, "composite", "fixed_cpu_upstream"),
        (device, "composite", "ctc"),
    ]:
        diagnosis.seed(target)
        model = diagnosis.model_for(arm, target)
        model.load_state_dict(saved["model"])
        if variant == "composite":
            for block in model.blocks:
                block.forward = types.MethodType(composite_block_forward, block)
        model.train()
        logits, lengths = model(*diagnosis.batch_for(rows, arm, target))
        logits.retain_grad()
        loss = ctc_objective(logits, lengths, targets)
        if backward == "ctc":
            loss.backward()
        else:
            logits.backward(reference_upstream.to(target))
        observed = {
            n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None
        }
        stats = diagnosis.gradients(model)
        if reference_grads is None:
            reference_grads = observed
            reference_logits = logits.detach().cpu()
            reference_upstream = logits.grad.detach().cpu()
            torch.save(
                {
                    "gradients": observed,
                    "logits": reference_logits,
                    "upstream": reference_upstream,
                    "diagnostic_only": True,
                },
                OUT / f"{arm}_cpu_gradient_reference.pt",
            )
        squared_diff = 0.0
        squared_reference = 0.0
        max_diff = 0.0
        per_parameter = []
        for name, grad in observed.items():
            reference = reference_grads[name].double()
            difference = grad.double() - reference
            error = float((difference**2).sum())
            denom = float((reference**2).sum())
            squared_diff += error
            squared_reference += denom
            maximum = float(difference.abs().max())
            max_diff = max(max_diff, maximum)
            per_parameter.append(
                {
                    "parameter": name,
                    "relative_l2_error": diagnosis.scalar((error / max(denom, 1e-30)) ** 0.5),
                    "max_abs_error": diagnosis.scalar(maximum),
                }
            )
        report = {
            "device": target,
            "variant": variant,
            "backward": backward,
            "loss": diagnosis.scalar(loss.detach()),
            "logits": diagnosis.tensor_stats(logits),
            "logits_max_abs_error_vs_cpu": float(
                (logits.detach().cpu() - reference_logits).abs().max()
            ),
            "nonfinite_parameters": stats["nonfinite_parameters"],
            "global_gradient_l2_float64": stats["global_l2_float64"],
            "global_gradient_relative_l2_error_vs_cpu": diagnosis.scalar(
                (squared_diff / max(squared_reference, 1e-30)) ** 0.5
            ),
            "gradient_max_abs_error_vs_cpu": diagnosis.scalar(max_diff),
            "per_parameter_errors": per_parameter,
        }
        reports.append(report)
        print(
            json.dumps({k: v for k, v in report.items() if k != "per_parameter_errors"}), flush=True
        )
        del model, logits, loss
        if target.startswith("xpu"):
            torch.xpu.empty_cache()
    diagnosis.original.protocol.freeze(
        OUT / f"{arm}_composite_full_model.json",
        {
            "arm": arm,
            "training_uids": saved["uids"],
            "reports": reports,
            "optimizer_updates": 0,
            "validation_used": False,
            "closed_p10_resumed": False,
            "production_module_changed": False,
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
