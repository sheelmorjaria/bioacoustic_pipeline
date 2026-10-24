"""Host-only audit of corrected scientific-runner preparation; never executes it."""

import json
from pathlib import Path
import torch
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write

ROOT = Path("outputs/rclat_rtpe_multiseed_v2_execution")


def main():
    manifest = json.loads((ROOT / "execution_manifest.json").read_text())
    assert sha(ROOT / "protocol.json") == manifest["protocol_sha256"]
    assert all(sha(p) == h for p, h in manifest["dependencies"].items())
    old = json.loads(Path("outputs/rclat_rtpe_multiseed_v1/protocol.json").read_text())
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert all(
        protocol[k] == v for k, v in old.items() if k not in {"study", "status", "frozen_utc"}
    )
    assert (ROOT / "protocol.json").read_bytes() == Path(
        "outputs/rclat_rtpe_multiseed_v2_spec/protocol.json"
    ).read_bytes()
    assert not any(
        (ROOT / n).exists()
        for n in [
            "AUTHORIZATION.json",
            "RUN_REQUESTED.json",
            "global_selection_lock.json",
            "result.json",
        ]
    )
    assert not list(ROOT.glob("seed_*"))
    preflight = json.loads((ROOT / "preflight.json").read_text())
    source = json.loads((ROOT / "input_verification.json").read_text())
    assert (
        preflight["passed"]
        and len(preflight["cases"]) == 15
        and source["records"] == 2814
        and source["passed"]
    )
    for obj in [preflight, source]:
        assert obj["execution_sha256"] == sha(ROOT / "execution_manifest.json")
    maximum = 0.0
    count = 0
    for p in ROOT.glob("preflight_seed_*/parity_tensors.pt"):
        tensors = torch.load(p, map_location="cpu", weights_only=True)
        for a, b in zip(tensors["CPU"], tensors["XPU"]):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            error = float((a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-30))
            assert error < 1e-4
            maximum = max(maximum, error)
        count += 1
    assert count == 15
    predecessors = {}
    for name in [
        "outputs/rclat_rtpe1_execution_v1/training/artifact_manifest.json",
        "outputs/rclat_rtpe_multiseed_v1/closeout_manifest.json",
    ]:
        d = json.loads(Path(name).read_text())
        assert all(sha(p) == h for p, h in d.items())
        predecessors[name] = len(d)
    for name in ["preregistration.json", "execution_manifest.json"]:
        p = Path("outputs/rclat_bilstm_dropout_v1") / name
        d = json.loads(p.read_text())["dependencies"]
        assert all(sha(p) == h for p, h in d.items())
        predecessors[str(p)] = len(d)
    audit = dict(
        status="CORRECTED_RUNNER_PREPARATION_AND_PREFLIGHT_PASSED_NOT_AUTHORIZED",
        CPU_tests=87,
        lint_passed=True,
        source_cache_pairs=2814,
        preflight_cases=count,
        maximum_relative_error=maximum,
        tolerance=1e-4,
        scientific_fields_preserved=True,
        execution_manifest_sha256=sha(ROOT / "execution_manifest.json"),
        verified_predecessor_dependencies=predecessors,
        scientific_updates=0,
        scientific_ridge_fits=0,
        internal_evaluation_scores=0,
        scientific_execution_authorized=False,
        replication_outcome="UNEVALUATED",
    )
    write(ROOT / "preparation_closeout.json", audit)
    text = f"""# RTPE-R1 v2 corrected runner preparation — PASSED

A dedicated corrected runner is implemented and separately pinned to the frozen successor specification. It binds the certified WHERE-guard model in numerical preflight, training, validation and selected/final checkpoint inference; the weighted training loss uses the guarded implementation. Ridge/scaling and host evaluation-loss arithmetic are unchanged.

- **87 CPU tests** and lint passed. Synthetic tests verify all-five training-before-evaluation ordering, deterministic sampler use, no retry after failure, authorisation rejection and guarded preflight semantics. AST comparisons verify that staged training, evaluation and lock function bodies match the frozen predecessor exactly; only model/loss import bindings differ.
- **2,814 source/cache pairs** were reverified. Source integrity checks did not compute evaluation predictions or scores.
- **15/15 registered XPU-0 numerical preflight cases** passed. Maximum relative error: **{maximum:.9g}**, below 1e-4. An independent host audit recomputed parity from saved tensors. Each case has a durable start/result and saved tensors; global preflight failures are retained and cannot be retried over an existing attempt.
- A separate execution manifest binds protocol, candidate guards, staging code, tests, fixtures and environment. Original scientific fields and seed pairs remain unchanged. Previous scientific artifact hashes and dropout dependencies were verified unchanged.

The CPU tests fit only explicitly synthetic test data. The real-training preflight performed forward/backward computations without optimiser updates. No scientific Ridge models were fitted, no scientific optimiser updates occurred and internal evaluation remained sealed.

Scientific execution is **not authorised or started**. The runner requires a separate explicit authorisation record bound to the exact protocol and execution-manifest hashes before its execute action. No such record has been created. There are no scientific seed directories, global selection lock or evaluation result.

Preparation commands used: `prepare`, `verify`, and the bounded 120-second `preflight` with the frozen cache policy. The execute command was not invoked. No correction to the already frozen successor specification was made. Its earlier runner-pending text is historical; this preparation closeout and the separate execution manifest record the completed implementation milestone.

The five-seed predictive-utility outcome remains UNEVALUATED. Original RTPE-1 remains a closed positive development result. P100/P960 and MACDE–CAVEM Stage 2 remain blocked. Full-run stability remains an execution uncertainty; short preflight success does not establish the scientific outcome.
"""
    with (ROOT / "RESULTS.md").open("x") as f:
        f.write(text)
    write(ROOT / "artifact_manifest.json", {str(p): sha(p) for p in ROOT.rglob("*") if p.is_file()})
    write(
        ROOT / "report_manifest.json",
        {
            str(Path(__file__)): sha(__file__),
            str(ROOT / "artifact_manifest.json"): sha(ROOT / "artifact_manifest.json"),
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
