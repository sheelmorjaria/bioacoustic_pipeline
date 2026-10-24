"""Summarise recorded native stacks; does not run a model or touch raw captures."""

from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.append("/tmp/rclat_memray")
import memray  # noqa: E402

root = Path("outputs/rclat_p10_shape_cycles")
summary = {}
categories = {}
for suffix, name in [("_first", "first_use"), ("", "repeated")]:
    path = root / f"cycling_xpu0_native_windows{suffix}.bin"
    reader = memray.FileReader(path)
    records = sorted(reader.get_leaked_allocation_records(), key=lambda r: r.size, reverse=True)
    row = dict(
        capture=str(path),
        retained_bytes=sum(r.size for r in records),
        retained_allocation_count=sum(r.n_allocations for r in records),
        retained_stack_groups=len(records),
        top=[],
    )
    totals = defaultdict(int)
    counts = defaultdict(int)
    for i, r in enumerate(records):
        frames = r.native_stack_trace()
        funcs = [f[0] for f in frames]
        if any("CompressedRTDeviceBinaryImage::Decompress" in f for f in funcs):
            key = "SYCL_device_image_decompression"
        elif any(
            "urProgramCreateWithBinary" in f or "ur_program_handle_t_::ur_program_handle_t_" in f
            for f in funcs
        ):
            key = "UR_program_binary_creation"
        elif any("XPUCachingAllocator" in f for f in funcs):
            key = "XPU_allocator_path"
        elif any("dnnl" in f.lower() for f in funcs):
            key = "oneDNN_path"
        else:
            key = "other_or_unresolved"
        totals[key] += r.size
        counts[key] += r.n_allocations
        if i < 15:
            row["top"].append(
                dict(
                    bytes=r.size,
                    allocations=r.n_allocations,
                    native_frames=[list(f) for f in frames[:25]],
                    python_frames=[list(f) for f in r.stack_trace()[:6]],
                )
            )
    summary[name] = row
    categories[name] = {
        k: dict(retained_bytes=v, allocation_count=counts[k]) for k, v in totals.items()
    }
(root / "native_stack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(root / "native_path_categories.json").write_text(json.dumps(categories, indent=2) + "\n")
print(
    "Native stack summaries written; categories are ordered stack-pattern matches, not exclusive subsystem ownership proofs."
)
