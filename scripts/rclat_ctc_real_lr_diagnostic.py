"""Real-speech LR-only comparison reusing the frozen D1 runner unchanged."""

import argparse
import json
import os
from pathlib import Path
import traceback

import rclat_ctc_learning_diagnostic as core
from rclat_ctc_lr_diagnostic import learning_rate

ROOT = Path("outputs/rclat_ctc_real_lr_diagnostic_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "d1"])
    parser.add_argument("--device", required=True, choices=["xpu:0", "xpu:1"])
    parser.add_argument("--policy", required=True, choices=["original", "capped250"])
    args = parser.parse_args()
    plan = json.loads((ROOT / "execution_plan.json").read_text())
    for path, digest in plan["dependencies"].items():
        if core.sha(path) != digest:
            raise ValueError("Changed dependency: " + path)
    if os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384":
        raise ValueError("Frozen cache policy missing")
    out = ROOT / args.policy / args.device.replace(":", "_")
    if json.loads((out / "execution_plan.json").read_text()) != {**plan, "device": args.device, "policy": args.policy}:
        raise ValueError("Case plan mismatch")
    core.OUT = out
    core.CHECKPOINTS = Path("checkpoints/rclat_ctc_real_lr_diagnostic_v1") / args.policy / args.device.replace(":", "_")
    core.predecessor.original.learning_rate = lambda step, total: learning_rate(step, args.policy)
    try:
        core.run(args.action, args.device)
    except Exception:
        failure = out / f"{args.action}_FAILURE.json"
        if not failure.exists():
            core.write(failure, {"traceback": traceback.format_exc(), "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
