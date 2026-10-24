"""Counterbalanced per-device execution; fail immediately without retries."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("outputs/rclat_ctc_lr_diagnostic_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=["xpu:0", "xpu:1"])
    args = parser.parse_args()
    plan = json.loads((ROOT / "execution_plan.json").read_text())
    env = {**os.environ, "ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16384", "ONEDNN_VERBOSE": "profile_create,filter=conv"}
    for policy in plan["execution_order"][args.device]:
        out = ROOT / policy / args.device.replace(":", "_")
        for action in ["preflight", "run"]:
            with (out / f"{action}_console.log").open("x") as stream:
                subprocess.run(
                    [sys.executable, "scripts/rclat_ctc_lr_diagnostic.py", action, "--device", args.device, "--policy", policy],
                    env=env, stdout=stream, stderr=subprocess.STDOUT, check=True,
                    timeout=330 if action == "preflight" else 1830,
                )
            print(args.device, policy, action, "completed", flush=True)


if __name__ == "__main__":
    main()
