"""Offline cache-policy diagnostic analysis; no model execution."""

from collections import defaultdict, Counter
import json
from pathlib import Path
import numpy as np

root = Path("outputs/rclat_p10_cache_policy")
summary = {}
for path in sorted(root.glob("*_run.json")):
    name = path.name.removesuffix("_run.json")
    r = json.loads(path.read_text())
    cycles = defaultdict(Counter)
    cycle = 0
    for line in (root / f"{name}_run.log").open():
        if line.startswith("CACHE_CYCLE_BEGIN:"):
            cycle = int(line.split(":")[1])
        elif "create:cache_" in line and ",gpu:" in line and ",convolution," in line:
            cycles[cycle]["hit" if "cache_hit" in line else "miss"] += 1
    ends = [s for s in r["snapshots"] if s["batch"] == 128]
    changes = [
        dict(
            from_cycle=a["cycle"],
            to_cycle=b["cycle"],
            native_in_use_MiB=(b["glibc_in_use"] - a["glibc_in_use"]) / 1024**2,
            RSS_MiB=(int(b["VmRSS"].split()[0]) - int(a["VmRSS"].split()[0])) / 1024,
            seconds=b["seconds"] - a["seconds"],
        )
        for a, b in zip(ends, ends[1:])
    ]
    summary[name] = dict(
        completed_updates=r["completed_updates"],
        stop_reason=r["stop_reason"],
        seconds=r["seconds"],
        cache_counts={str(k): dict(v) for k, v in cycles.items()},
        cycle_changes=changes,
        primitive_failure=r["primitive_failure"],
    )
(root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
vectors = {}
for d in [0, 1]:
    a = root / f"default_xpu{d}_engineering_final_vector.npy"
    b = root / f"capacity8192_xpu{d}_engineering_final_vector.npy"
    if (
        a.exists()
        and b.exists()
        and (root / f"default_xpu{d}_run.json").exists()
        and (root / f"capacity8192_xpu{d}_run.json").exists()
    ):
        x = np.load(a).astype(np.float64)
        y = np.load(b).astype(np.float64)
        vectors[f"xpu{d}"] = dict(
            bit_equal=bool(np.array_equal(x, y)),
            relative_L2=float(np.linalg.norm(x - y) / np.linalg.norm(x)),
            maximum_absolute_difference=float(np.max(np.abs(x - y))),
        )
(root / "final_vector_comparison.json").write_text(json.dumps(vectors, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("FINAL_VECTORS", vectors)
