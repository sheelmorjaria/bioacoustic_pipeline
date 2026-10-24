"""Bounded scientific launch; no retry, restart or resumption."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path("outputs/rclat_bilstm_dropout_v1")
parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["xpu:0", "xpu:1"], required=True)
a = parser.parse_args()
out = ROOT / a.device.replace(":", "_")
spec = json.loads((ROOT / "preregistration.json").read_text())
env = {
    **os.environ,
    "ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16384",
    "ONEDNN_VERBOSE": "profile_create,filter=conv",
}
try:
    with (out / "training_console.log").open("x") as stream:
        subprocess.run(
            [
                sys.executable,
                "scripts/rclat_bilstm_dropout_train.py",
                "train",
                "--device",
                a.device,
            ],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=spec["training"]["external_walltime_seconds_per_device"],
        )
except Exception as exc:
    path = out / "launcher_FAILURE.json"
    if not path.exists():
        path.write_text(json.dumps({"error": repr(exc), "automatic_retry": False}, indent=2) + "\n")
    raise
print(a.device, "scientific execution completed", flush=True)
