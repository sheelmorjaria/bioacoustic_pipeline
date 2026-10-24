"""Frozen LR-only intervention using the unchanged synthetic diagnostic runner."""

import argparse
import json
import os
from pathlib import Path

import rclat_ctc_synthetic_diagnostic as source

ROOT = Path("outputs/rclat_ctc_lr_diagnostic_v1")
ORIGINAL_RATE = source.core.predecessor.original.learning_rate


def learning_rate(step, policy):
    rate = ORIGINAL_RATE(step, 21120)
    if policy == "original":
        return rate
    if policy == "capped250":
        return min(rate, ORIGINAL_RATE(250, 21120))
    raise ValueError("Unknown learning-rate policy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "run"])
    parser.add_argument("--device", required=True, choices=["xpu:0", "xpu:1"])
    parser.add_argument("--policy", required=True, choices=["original", "capped250"])
    args = parser.parse_args()
    plan = json.loads((ROOT / "execution_plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if source.core.sha(path) != digest:
            raise ValueError("Changed dependency: " + path)
    if os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384":
        raise ValueError("Frozen cache policy missing")
    _, design, cohort = source.verify()
    out = ROOT / args.policy / args.device.replace(":", "_")
    case_plan = {**plan, "device": args.device, "policy": args.policy}
    if json.loads((out / "execution_plan.json").read_text()) != case_plan:
        raise ValueError("Case plan mismatch")
    source.OUT = out
    source.CHECKPOINTS = Path("checkpoints/rclat_ctc_lr_diagnostic_v1") / args.policy / args.device.replace(":", "_")
    source.core.OUT = out
    source.core.CHECKPOINTS = source.CHECKPOINTS
    source.verify = lambda: (case_plan, design, cohort)
    source.core.predecessor.original.learning_rate = lambda step, total: learning_rate(step, args.policy)
    try:
        source.run(args.action, args.device)
    except Exception:
        import traceback

        failure = out / f"{args.action}_FAILURE.json"
        if not failure.exists():
            source.core.write(failure, {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
