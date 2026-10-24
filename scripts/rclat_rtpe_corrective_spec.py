"""Freeze a corrective specification and preparation manifest; cannot train."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import numpy as np
import torch
from bioacoustic_pipeline.diagnostics.rclat_rtpe_corrective_spec import corrective_protocol
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write

ROOT = Path("outputs/rclat_rtpe_multiseed_v2_spec")


def main():
    predecessor = Path("outputs/rclat_rtpe_multiseed_v1/protocol.json")
    certificate = Path("outputs/rclat_rtpe_guard_execution_v1/certificate.json")
    assert (
        json.loads(certificate.read_text())["status"]
        == "BOUNDED_TWO_DEVICE_GUARD_AND_OPTIMISER_CHECKS_PASSED"
    )
    manifests = [
        "outputs/rclat_rtpe_guard_execution_v1/manifest.json",
        "outputs/rclat_rtpe_guard_execution_v1/report_manifest.json",
        "outputs/rclat_rtpe_multiseed_v1/closeout_manifest.json",
    ]
    for name in manifests:
        assert all(sha(p) == h for p, h in json.loads(Path(name).read_text()).items())
    original = json.loads(predecessor.read_text())
    protocol = corrective_protocol(
        original, datetime.now(timezone.utc).isoformat(), sha(predecessor)
    )
    for key, value in original.items():
        if key not in {"study", "status", "frozen_utc"}:
            assert protocol[key] == value
    ROOT.mkdir(exist_ok=False)
    write(ROOT / "protocol.json", protocol)
    dependencies = [
        predecessor,
        certificate,
        Path(__file__),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_corrective_spec.py"),
        Path("tests/test_rclat_rtpe_corrective_spec.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_guarded.py"),
    ] + [Path(p) for p in manifests]
    write(
        ROOT / "preparation_manifest.json",
        dict(
            status="SPECIFICATION_ONLY_NOT_EXECUTION_MANIFEST",
            protocol_sha256=sha(ROOT / "protocol.json"),
            dependencies={str(p): sha(p) for p in dependencies},
            environment=dict(
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__
            ),
            scientific_execution_authorized=False,
            scientific_updates=0,
            evaluation_scores=0,
        ),
    )
    (ROOT / "README.md").write_text("""# RTPE-R1 corrective successor specification v2

The scientific specification is locally frozen; this is not external registration or execution authorisation. All predecessor scientific fields are preserved verbatim except study/status/freeze metadata. The added correction specifies only the certified WHERE-based input and loss guards.

The original five initialization/sampler pairs, XPU-0 sequential assignment, CPU Ridge grid, architecture, origins, masks, weights, optimizer policy, 5,000 updates per seed, selection rules, 80 ms endpoint, 10,000-draw speaker intervals and all-five decision rule are unchanged. All 25,000 updates and five validation selections must precede any new evaluation scoring. Historical seed 17 is excluded from the replication gate. No engineering checkpoint may be reused.

The failed predecessor is preserved as preflight-failed and scientifically unevaluated. This successor is a new specification, not a retry or continuation. The preparation manifest is not an execution manifest.

Next: bind and verify a dedicated corrected runner, including its preflight and evaluation lock; pin a separate execution manifest and verify sources/environment. Scientific execution requires explicit authorisation. This preparation script has no train, preflight or evaluate command. No model was fitted or scored.

Original RTPE-1 remains closed and positive. Five-seed predictive utility remains unevaluated. P100/P960 and MACDE–CAVEM Stage 2 remain blocked.
""")
    write(ROOT / "manifest.json", {str(p): sha(p) for p in ROOT.iterdir() if p.is_file()})
    print(
        "Corrective scientific specification frozen;runner binding pending;no training or scoring."
    )


if __name__ == "__main__":
    main()
