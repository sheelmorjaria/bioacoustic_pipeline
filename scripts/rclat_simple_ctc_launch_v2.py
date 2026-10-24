"""Run one frozen encoder-control preflight and fitting probe; no retries."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True, choices=["xpu:0", "xpu:1"])
a = parser.parse_args()
out = Path("outputs/rclat_simple_ctc_diagnostic_v2") / a.device.replace(":", "_")
env = {**os.environ, "ONEDNN_PRIMITIVE_CACHE_CAPACITY":"16384", "ONEDNN_VERBOSE":"profile_create,filter=conv"}
for action in ["preflight", "d1"]:
    with (out / f"{action}_console.log").open("x") as stream:
        subprocess.run([sys.executable, "scripts/rclat_simple_ctc_diagnostic_v2.py", action, "--device", a.device], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=330 if action == "preflight" else 3660)
    print(a.device, action, "completed", flush=True)
