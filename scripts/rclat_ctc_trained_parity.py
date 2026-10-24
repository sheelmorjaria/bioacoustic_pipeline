"""Frozen-state CPU/XPU forward and gradient comparison; zero updates."""

import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import torch
import rclat_ctc_synthetic_diagnostic as source

OUT = Path("outputs/rclat_ctc_trained_parity_v1")


def relative(a, b):
    x = a.detach().cpu().double()
    y = b.detach().cpu().double()
    return float((x - y).norm() / x.norm().clamp_min(1e-300))


def run(device):
    plan = json.loads((OUT / "plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if source.core.sha(path) != digest:
            raise ValueError("Pinned dependency changed: " + path)
    _, _, cohort = source.verify()
    for entry in plan["checkpoints"].values():
        if source.core.sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Checkpoint content changed")
    if device not in plan["devices"]:
        raise ValueError("Device changed")
    name = device.replace(":", "_")
    source.core.write(
        OUT / f"{name}_STARTED.json",
        dict(device=device, plan_sha256=source.core.sha(OUT / "plan.json"), optimizer_updates=0),
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.xpu.manual_seed_all(0)
    torch.xpu.set_device(device)
    byuid = {r["uid"]: r for r in cohort["training"]}
    batch = source.batcher(device)
    cases = []
    started = time.monotonic()
    with (OUT / f"{name}_progress.jsonl").open("x", buffering=1) as journal:
        for step, entry in plan["checkpoints"].items():
            saved = torch.load(entry["path"], map_location="cpu", weights_only=True)
            cpu = source.core.build()
            cpu.load_state_dict(saved["model"], strict=True)
            cpu.train()
            gpu = copy.deepcopy(cpu).to(device)
            before = source.core.predecessor.initial_digest(cpu)
            del saved
            for index, uids in enumerate(plan["batches"]):
                if time.monotonic() - started > plan["seconds_bound"]:
                    raise TimeoutError("Parity diagnostic bound")
                cpu.zero_grad(set_to_none=True)
                gpu.zero_grad(set_to_none=True)
                rows = [byuid[uid] for uid in uids]
                inputs, targets = batch(rows, "cpu")
                yc, lc = cpu(*inputs)
                source.core.assert_lengths(yc, lc, inputs[2], targets)
                if not bool(torch.isfinite(yc).all()):
                    raise FloatingPointError("Nonfinite CPU logits")
                yc.retain_grad()
                loss_c = source.core.ctc_objective(yc, lc, targets)
                loss_c.backward()
                source.core.check_gradients(cpu)
                yg, lg = gpu(inputs[0].to(device), inputs[1].to(device), inputs[2])
                source.core.assert_lengths(yg, lg, inputs[2], targets)
                if not bool(torch.isfinite(yg).all()):
                    raise FloatingPointError("Nonfinite XPU logits")
                yg.retain_grad()
                loss_g = source.core.ctc_objective(yg, lg, targets)
                loss_g.backward()
                source.core.check_gradients(gpu)
                if not torch.equal(lc, lg):
                    raise ValueError("Output lengths differ")
                numerator = denominator = 0.0
                parameters = []
                for (n, p), (m, q) in zip(cpu.named_parameters(), gpu.named_parameters()):
                    if n != m:
                        raise ValueError("Parameter schema mismatch")
                    a = p.grad.detach().double()
                    b = q.grad.detach().cpu().double()
                    num = float((a - b).square().sum())
                    den = float(a.square().sum())
                    numerator += num
                    denominator += den
                    parameters.append(
                        dict(
                            name=n,
                            reference_norm=den**0.5,
                            absolute_L2=num**0.5,
                            relative_L2=num**0.5 / max(den**0.5, 1e-12),
                            max_absolute_error=float((a - b).abs().max()),
                        )
                    )
                gradient = (numerator / max(denominator, 1e-300)) ** 0.5
                valid = torch.arange(yc.shape[1])[None, :] < lc[:, None]
                logits = relative(yc[valid], yg.detach().cpu()[valid])
                loss = abs(float(loss_c.detach()) - float(loss_g.detach())) / max(
                    abs(float(loss_c.detach())), 1e-30
                )
                logits_gradient = relative(yc.grad, yg.grad)
                # Secondary check of the installed CPU CTC path at the saved logits.
                double_logits = yc.detach().double().requires_grad_(True)
                double_loss = source.core.ctc_objective(double_logits, lc, targets)
                double_grad = torch.autograd.grad(double_loss, double_logits)[0]
                ctc_precision_gradient = relative(double_grad, yc.grad)
                ctc_precision_loss = abs(
                    float(double_loss.detach()) - float(loss_c.detach())
                ) / max(abs(float(double_loss.detach())), 1e-30)
                passed = max(gradient, logits, loss, logits_gradient) <= plan["tolerance"]
                ctc_passed = max(ctc_precision_gradient, ctc_precision_loss) <= plan["tolerance"]
                unchanged = (
                    before
                    == source.core.predecessor.initial_digest(cpu)
                    == source.core.predecessor.initial_digest(gpu)
                )
                if not unchanged:
                    raise ValueError("Model parameters changed during no-update check")
                row = dict(
                    checkpoint_step=int(step),
                    batch=index,
                    uids=uids,
                    gradient_relative_L2=gradient,
                    valid_logit_relative_L2=logits,
                    loss_relative=loss,
                    logit_gradient_relative_L2=logits_gradient,
                    CPU_FP32_vs_FP64_logit_gradient_relative_L2=ctc_precision_gradient,
                    CPU_FP32_vs_FP64_loss_relative=ctc_precision_loss,
                    passed=passed,
                    CPU_CTC_precision_check_passed=ctc_passed,
                    model_parameters_unchanged=unchanged,
                    reference_gradient_norm=denominator**0.5,
                    per_parameter=parameters,
                )
                cases.append(row)
                journal.write(
                    json.dumps({k: v for k, v in row.items() if k != "per_parameter"}) + "\n"
                )
                print(
                    json.dumps({k: v for k, v in row.items() if k != "per_parameter"}), flush=True
                )
                del inputs, targets, yc, yg, loss_c, loss_g, double_logits, double_loss, double_grad
            del cpu, gpu
    source.core.write(
        OUT / f"{name}_result.json",
        dict(
            completed=True,
            cases=cases,
            all_primary_checks_passed=all(r["passed"] for r in cases),
            all_secondary_CPU_precision_checks_passed=all(
                r["CPU_CTC_precision_check_passed"] for r in cases
            ),
            elapsed_seconds=time.monotonic() - started,
            optimizer_updates=0,
            checkpoints_modified=False,
            stage2_ready=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["xpu:0", "xpu:1"], required=True)
    a = parser.parse_args()
    try:
        run(a.device)
    except Exception:
        path = OUT / f"{a.device.replace(':', '_')}_FAILURE.json"
        if not path.exists():
            source.core.write(path, dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
