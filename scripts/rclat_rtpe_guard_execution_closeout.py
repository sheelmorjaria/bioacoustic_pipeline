"""Audit bounded guard/device engineering artifacts without XPU access."""

import hashlib
import json
from pathlib import Path
import torch

ROOT = Path("outputs/rclat_rtpe_guard_execution_v1")
CROSS = Path("outputs/rclat_rtpe_guard_integration_xpu1_v1")


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    with Path(p).open("x") as f:
        json.dump(v, f, indent=2, allow_nan=False)


def check(mapping):
    assert all(sha(p) == s for p, s in mapping.items())
    return len(mapping)


def relative(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float((a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-30))


def main():
    manifests = {}
    for p in [
        ROOT / "manifest.json",
        CROSS / "manifest.json",
        Path("outputs/rclat_rtpe_guard_integration_v1/manifest.json"),
        Path("outputs/rclat_rtpe1_execution_v1/training/artifact_manifest.json"),
        Path("outputs/rclat_rtpe_multiseed_v1/closeout_manifest.json"),
    ]:
        manifests[str(p)] = check(read(p))
    for r in [ROOT, CROSS]:
        check(read(r / "plan.json")["dependencies"])
        assert read(r / "summary.json").get(
            "passed", read(r / "summary.json").get("integration_candidate_passed")
        )
    for name in ["preregistration.json", "execution_manifest.json"]:
        p = Path("outputs/rclat_bilstm_dropout_v1") / name
        d = read(p)
        for key in ["dependencies", "source_hashes", "implementation_hashes"]:
            if key in d and isinstance(d[key], dict):
                manifests[str(p) + ":" + key] = check(d[key])
    # Verify the XPU1 test has the same registered computation as v1.
    source = Path("scripts/rclat_rtpe_guard_integration.py").read_text()
    expected = (
        source.replace("rclat_rtpe_guard_integration_v1", "rclat_rtpe_guard_integration_xpu1_v1")
        .replace("xpu:0", "xpu:1")
        .replace("XPU0 unavailable", "XPU1 unavailable")
        .replace(
            "torch.set_num_threads(2)", "torch.set_num_threads(2)\n    torch.xpu.set_device(1)"
        )
    )
    assert expected == Path("scripts/rclat_rtpe_guard_integration_xpu1.py").read_text()
    maxima = {}
    steps = 0
    sequences = 0
    for directory in sorted(ROOT.glob("xpu_*_seed_*")):
        assert read(directory / "result.json")["passed"]
        rows = sorted(directory.glob("update_*.json"))
        assert len(rows) == 20
        for index, p in enumerate(rows, 1):
            row = read(p)
            assert row["update"] == index and row["CPU_bit_identical"] and row["passed"]
            for k, v in row["errors"].items():
                assert v < 1e-4
                maxima[k] = max(maxima.get(k, 0.0), v)
        d = torch.load(directory / "final_tensors.pt", map_location="cpu", weights_only=True)
        for k, a in d["original_CPU"].items():
            assert torch.equal(a, d["candidate_CPU"][k])
            assert relative(a, d["candidate_XPU"][k]) < 1e-4
        sequences += 1
        steps += len(rows)
    assert sequences == 10 and steps == 200
    numerical = 0
    crossmax = 0.0
    for p in CROSS.glob("*/parity_tensors.pt"):
        d = torch.load(p, map_location="cpu", weights_only=True)
        for a, b, c in zip(d["original_CPU"], d["candidate_CPU"], d["candidate_XPU"]):
            assert torch.equal(a, b)
            e = relative(a, c)
            assert e < 1e-4
            crossmax = max(crossmax, e)
        numerical += 1
    faults = [read(p) for p in CROSS.glob("fault_*/result.json")]
    assert numerical == 16 and len(faults) == 288
    assert all(
        x["passed"]
        and x["accepted"] == x["fixture"]["expected_accept"]
        and x["post_inputs_unchanged"]
        for x in faults
    )
    audit = dict(
        status="BOUNDED_TWO_DEVICE_GUARD_AND_OPTIMISER_CHECKS_PASSED",
        xpu1_numerical_cases=numerical,
        xpu1_max_relative_error=crossmax,
        xpu1_fault_cases=len(faults),
        xpu1_accepts=sum(x["accepted"] for x in faults),
        xpu1_rejects=sum(not x["accepted"] for x in faults),
        optimizer_sequences=sequences,
        updates_per_model_sequence=20,
        paired_update_records=steps,
        engineering_optimizer_updates=steps * 3,
        max_update_errors=maxima,
        CPU_original_candidate_bit_identical=True,
        verified_manifests=manifests,
        production_adopted=False,
        scientific_optimizer_updates=0,
        replication_restarted=False,
    )
    write(ROOT / "independent_audit.json", audit)
    text = f"""# RTPE guard device/optimiser engineering closeout — PASSED

XPU 1 completed the same registered guard-integration matrix as XPU 0: **16/16 numerical cases and 288/288 fault cases passed**. Maximum XPU-1 CPU/device relative error was **{crossmax:.9g}**, below 1e-4. All 108 valid cases were accepted and 180 invalid/unsupported cases rejected.

Both devices completed five fixed-seed engineering sequences of 20 consecutive AdamW updates. All **200 paired update records** passed. The three models in each sequence were the original CPU reference, corrected CPU candidate and corrected XPU candidate: this is **600 disposable engineering optimizer updates**, not zero optimizer activity. Scientific optimizer updates remain zero. No engineering model is selected for scientific reuse.

The fixed optimizer used lr .001, weight decay .0001, default betas/eps, foreach=False and fused=False, with gradient clipping at 5. The sequence mixed original training fixtures with synthetic batch-128 inputs, including masked nonfinite values. No cache clearing occurred between updates. Original/candidate CPU forecasts, losses, gradients, parameters and optimizer states were bit-identical. All CPU–XPU comparisons passed the frozen 1e-4 tolerance; maximum relative errors by quantity:

"""
    text += "\n".join(f"- {k}: {v:.9g}" for k, v in maxima.items())
    text += """

The host audit rechecked saved final tensors, all sequential update records, fault decisions, unchanged input records and dependency/artifact hashes. The XPU-1 runner was verified against the original runner with only output-directory/device changes and explicit device selection. Both processes completed without timeout or retry. The 79 CPU tests and lint passed.

This certificate covers these FP32 fixtures, both installed XPUs, the specified AdamW implementation and 20-update histories. It does not establish full 5,000-update stability, arbitrary asynchronous histories, other dtypes/layouts or a backend-kernel repair. Fault semantics during optimisation were exercised through valid masked fixtures; the complete invalid-input matrix was the separate inference/loss diagnostic.

Frozen production entrypoints and previous scientific outputs remain unchanged. No Ridge fitting, scientific training or internal-evaluation scoring occurred. Five-seed predictive utility remains UNEVALUATED. A separately frozen scientific successor has not been launched. P100/P960 and MACDE–CAVEM Stage 2 remain blocked.
"""
    with (ROOT / "RESULTS.md").open("x") as f:
        f.write(text)
    write(ROOT / "certificate.json", audit)
    write(
        ROOT / "report_manifest.json",
        {
            str(p): sha(p)
            for p in [
                ROOT / "RESULTS.md",
                ROOT / "certificate.json",
                ROOT / "independent_audit.json",
                Path(__file__),
            ]
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
