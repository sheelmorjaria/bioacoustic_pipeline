"""Analyse completed working-set diagnostics and optional late native capture."""

from collections import defaultdict
import json
from pathlib import Path
import sys

root = Path("outputs/rclat_p10_large_shape_set")
comparison = {}
for path in sorted(root.glob("*_result.json")):
    data = json.loads(path.read_text())
    ends = [r for r in data["snapshots"] if r["batch"] == 128]
    changes = []
    for a, b in zip(ends, ends[1:]):
        changes.append(
            dict(
                from_cycle=a["cycle"],
                to_cycle=b["cycle"],
                native_in_use_MiB=(b["glibc_in_use"] - a["glibc_in_use"]) / 1024**2,
                RSS_MiB=(int(b["VmRSS"].split()[0]) - int(a["VmRSS"].split()[0])) / 1024,
            )
        )
    comparison[path.stem] = dict(
        completed_updates=data["completed_updates"],
        stop_reason=data["stop_reason"],
        historical_failure_reproduced=data["historical_primitive_failure_reproduced"],
        cycle_changes=changes,
    )
(root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
profile = root / "large128_xpu0_profiled.bin"
if profile.exists() and (root / "large128_xpu0_profiled_result.json").exists():
    sys.path.append("/tmp/rclat_memray")
    import memray

    reader = memray.FileReader(profile)
    records = sorted(reader.get_leaked_allocation_records(), key=lambda r: r.size, reverse=True)
    summary = dict(
        retained_bytes=sum(r.size for r in records),
        allocation_count=sum(r.n_allocations for r in records),
        stack_groups=len(records),
        top=[],
    )
    totals = defaultdict(int)
    counts = defaultdict(int)
    for i, r in enumerate(records):
        frames = r.native_stack_trace()
        names = [f[0] for f in frames]
        if any("CompressedRTDeviceBinaryImage::Decompress" in f for f in names):
            key = "SYCL_image_decompression"
        elif any(
            "urProgramCreateWithBinary" in f or "ur_program_handle_t_::ur_program_handle_t_" in f
            for f in names
        ):
            key = "UR_program_binary_creation"
        elif any("XPUCachingAllocator" in f for f in names):
            key = "XPU_allocator_path"
        elif any("dnnl" in f.lower() for f in names):
            key = "oneDNN_path"
        else:
            key = "other_or_unresolved"
        totals[key] += r.size
        counts[key] += r.n_allocations
        if i < 20:
            summary["top"].append(
                dict(
                    bytes=r.size,
                    allocations=r.n_allocations,
                    native_frames=[list(f) for f in frames[:40]],
                    python_frames=[list(f) for f in r.stack_trace()[:6]],
                )
            )
    summary["ordered_stack_pattern_categories"] = {
        k: dict(retained_bytes=v, allocations=counts[k]) for k, v in totals.items()
    }
    (root / "late_native_stack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(comparison, indent=2))

if profile.exists() and (root / "large128_xpu0_profiled_result.json").exists():
    detail = dict(
        total_bytes=0,
        oneDNN_path_bytes=0,
        JIT_kernel_creation_bytes=0,
        convolution_backward_path_bytes=0,
        reorder_JIT_path_bytes=0,
        categories_overlap=True,
    )
    for record in records:
        names = [f[0] for f in record.native_stack_trace()]
        detail["total_bytes"] += record.size
        if any("dnnl" in f.lower() for f in names):
            detail["oneDNN_path_bytes"] += record.size
        if any(
            "ngen::SYCLCodeGenerator" in f or ("dnnl" in f and "create_kernel" in f) for f in names
        ):
            detail["JIT_kernel_creation_bytes"] += record.size
        if any("convolution_backward" in f or "gen_bwd_weights_t" in f for f in names):
            detail["convolution_backward_path_bytes"] += record.size
        if any("reorder::jit" in f for f in names):
            detail["reorder_JIT_path_bytes"] += record.size
    detail["oneDNN_path_fraction"] = detail["oneDNN_path_bytes"] / detail["total_bytes"]
    (root / "late_native_path_detail.json").write_text(json.dumps(detail, indent=2) + "\n")

python_capture = root / "large128_xpu1_python_profiled.bin"
if python_capture.exists() and (root / "large128_xpu1_python_profiled_result.json").exists():
    sys.path.append("/tmp/rclat_memray")
    import memray

    python_reader = memray.FileReader(python_capture)
    rows = sorted(python_reader.get_leaked_allocation_records(), key=lambda r: r.size, reverse=True)
    result = dict(
        retained_bytes=sum(r.size for r in rows),
        allocation_count=sum(r.n_allocations for r in rows),
        stack_groups=len(rows),
        bytes_without_python_frames=sum(r.size for r in rows if not r.stack_trace()),
        top=[
            dict(
                bytes=r.size,
                allocations=r.n_allocations,
                python_frames=[list(f) for f in r.stack_trace()[:12]],
            )
            for r in rows[:15]
        ],
    )
    (root / "late_python_callsite_summary.json").write_text(json.dumps(result, indent=2) + "\n")
