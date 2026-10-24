"""Dedicated RTPE-1 revision-2 runner; prepare/verify never train or score."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import torch

from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import (
    execute_comparison,
    load_dataset,
    sha,
    write,
)
from bioacoustic_pipeline.representation.rclat_rtpe import validate_group_split

PROTOCOL = Path("outputs/rclat_rtpe1_protocol_v2/protocol.json")
ROOT = Path("outputs/rclat_rtpe1_execution_v1")
EXPECTED_PROTOCOL_SHA256 = "d54ca562fad6f22e54f696cdb949befa64bf218b63048b9a1c427fa605dd11f0"


def frozen():
    if sha(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Frozen protocol checksum differs from pinned revision2")
    p = json.loads(PROTOCOL.read_text())
    if (
        p["revision"] != 2
        or p["status"] != "LOCALLY_FROZEN_SCIENTIFIC_SPECIFICATION_NOT_EXTERNAL_REGISTRATION"
    ):
        raise ValueError("Only frozen revision2 is executable")
    for mapping in [p["dependencies"], p["preflight_artifacts"]]:
        for path, digest in mapping.items():
            if sha(path) != digest:
                raise ValueError("Frozen dependency changed: " + path)
    return p


def bindings():
    p = frozen()
    e = json.loads((ROOT / "execution_manifest.json").read_text())
    if e["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Protocol/manifest mismatch")
    for path, digest in e["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("Execution dependency changed: " + path)
    if e["environment"] != dict(
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__
    ):
        raise ValueError("Execution environment version changed")
    receipt_path = ROOT / "input_verification.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["execution_sha256"] != sha(ROOT / "execution_manifest.json") or receipt[
            "protocol_sha256"
        ] != sha(PROTOCOL):
            raise ValueError("Verified execution manifest or protocol changed")
    return p, e


def prepare():
    p = frozen()
    ROOT.mkdir(parents=True, exist_ok=False)
    paths = [
        PROTOCOL,
        Path(__file__),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_execution.py"),
        Path("tests/test_rclat_rtpe_execution.py"),
        Path("tests/test_rclat_rtpe_entrypoint.py"),
        PROTOCOL.parent / "manifest.json",
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_protocol.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_weighted.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_pool.py"),
    ]
    write(
        ROOT / "execution_manifest.json",
        dict(
            study=p["study"],
            protocol=str(PROTOCOL),
            protocol_sha256=sha(PROTOCOL),
            created_utc=datetime.now(timezone.utc).isoformat(),
            dependencies={str(f): sha(f) for f in paths},
            environment=dict(
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__
            ),
            entrypoint="scripts/rclat_rtpe_train.py run",
            device="xpu:0",
            scales=[1, 4, 16],
            sampler_seed=2026102301,
            torch_cpu_threads=2,
            inactive_heads="Order readout stays in state but requires_gradFalse and is excluded from optimizer; no symbols or behavioural head",
            admission="Do not start while a dropout training process is active; no interference with existing experiments",
            dependency_guards="Before run, each Ridge candidate, GRU checkpoint and final scoring",
            selected_checkpoint_integrity="Every checkpoint SHA256 verified before sealed evaluation;no resume",
            preparation_only=True,
            scientific_training_started=False,
            automatic_start=False,
            final_test_access=False,
            P100_authorized=False,
            P960_authorized=False,
            stage2_ready=False,
        ),
    )
    print("Execution manifest prepared. No fitting or scoring.", flush=True)


def verify_sources():
    p, e = bindings()
    if (ROOT / "RUN_REQUESTED.json").exists():
        raise RuntimeError("An execution attempt already exists; no re-verification/retry over it")
    records = json.loads(Path(p["inventory"]).read_text())
    validate_group_split(
        [dict(recording=r["uid"], individual=r["speaker"], split=r["split"]) for r in records]
    )
    if len(records) != 2814 or len({r["uid"] for r in records}) != 2814:
        raise ValueError("Frozen recording population mismatch")
    summaries = {}
    for split, expected in [
        ("train", (175, 1980, 15840)),
        ("validation", (38, 415, 3320)),
        ("test", (38, 419, 3352)),
    ]:
        rows = [r for r in records if r["split"] == split]
        counts = Counter(r["speaker"] for r in rows)
        actual = (len(counts), len(rows), sum(len(r["selected_origins"]) for r in rows))
        if actual != expected:
            raise ValueError("Frozen partition counts mismatch")
        for r in rows:
            if len(r["selected_origins"]) != 8 or len(set(r["selected_origins"])) != 8:
                raise ValueError("Eight unique frozen origins required")
            weight = 1 / (len(counts) * counts[r["speaker"]] * 8)
            if not np.isclose(r["origin_weight_within_split"], weight, rtol=0, atol=1e-14):
                raise ValueError("Frozen hierarchical weight mismatch")
        summaries[split] = dict(speakers=actual[0], recordings=actual[1], origins=actual[2])
    start = time.monotonic()
    for i, r in enumerate(records):
        if sha(r["source_path"]) != r["source_sha256"] or sha(r["cache_path"]) != r["cache_sha256"]:
            raise ValueError("Frozen waveform/cache mismatch: " + r["uid"])
        with np.load(r["cache_path"], allow_pickle=False) as f:
            mel = f["mel"]
            mask = f["mel_valid"]
            ends = f["frame_end_samples"]
        if (
            mel.ndim != 2
            or mel.shape[1] != 64
            or mel.dtype != np.float32
            or mask.shape != mel.shape
            or mask.dtype != np.dtype(bool)
            or not np.isfinite(mel[mask]).all()
            or not np.array_equal(ends, np.arange(len(mel)) * 160 + 640)
        ):
            raise ValueError("Frozen mel/timing schema mismatch")
        targets = np.array(r["selected_origins"])[:, None] + np.array(p["horizons"])[None]
        if hashlib_bytes(mask[targets].tobytes()) != r["selected_target_mask_sha256"]:
            raise ValueError("Frozen target support changed")
        for t in r["selected_origins"]:
            if t < 31 or t + 16 >= len(mel) or not mask[t - 31 : t + 17].any(1).all():
                raise ValueError("Frozen common origin became ineligible")
        if (i + 1) % 500 == 0:
            print(f"Verified {i + 1}/2814 waveform/cache pairs", flush=True)
    receipt = dict(
        passed=True,
        protocol_sha256=sha(PROTOCOL),
        execution_sha256=sha(ROOT / "execution_manifest.json"),
        source_cache_entries_verified=2814,
        partitions=summaries,
        elapsed_seconds=time.monotonic() - start,
        verification="Current compressed waveform/cache hashes bind previously verified PCM and regenerated features; exact sampled target masks and timing rechecked",
        scientific_models_fitted=0,
        internal_evaluation_scores_computed=0,
    )
    if (ROOT / "input_verification.json").exists():
        old = json.loads((ROOT / "input_verification.json").read_text())
        if (
            old["execution_sha256"] != receipt["execution_sha256"]
            or old["protocol_sha256"] != receipt["protocol_sha256"]
        ):
            raise ValueError("Existing verification receipt differs")
    else:
        write(ROOT / "input_verification.json", receipt)
    print("All2814waveform/cache pairs verified; no training or predictive scores.", flush=True)


def hashlib_bytes(value):
    import hashlib

    return hashlib.sha256(value).hexdigest()


def run():
    p, e = bindings()
    if (ROOT / "RUN_REQUESTED.json").exists() or (ROOT / "training").exists():
        raise RuntimeError("Existing scientific attempt preserved; no retry/resume")
    verified = json.loads((ROOT / "input_verification.json").read_text())
    if not verified["passed"] or verified["execution_sha256"] != sha(
        ROOT / "execution_manifest.json"
    ):
        raise ValueError("Matching input verification required")
    # Admission occurs before the irreversible scientific start marker.
    active = subprocess.check_output(["ps", "-eo", "args="], text=True)
    if any("rclat_bilstm_dropout_train.py" in line for line in active.splitlines()):
        raise RuntimeError("XPU admission deferred: dropout training remains active")
    if (
        os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384"
        or "DNNL_PRIMITIVE_CACHE_CAPACITY" in os.environ
    ):
        raise ValueError("Required requested cache policy absent or conflicting")
    if not torch.xpu.is_available():
        raise RuntimeError("Registered XPU0 unavailable")
    verify_sources()  # Recheck all frozen source/cache hashes immediately before start.
    start = time.monotonic()
    write(
        ROOT / "RUN_REQUESTED.json",
        dict(
            protocol_sha256=sha(PROTOCOL),
            execution_sha256=sha(ROOT / "execution_manifest.json"),
            started_utc=datetime.now(timezone.utc).isoformat(),
            scientific_training_started=True,
        ),
    )
    try:
        rows = json.loads(Path(p["inventory"]).read_text())
        train = load_dataset(
            rows, split="train", history=p["history"], horizons=p["horizons"], selection=None
        )
        validation = load_dataset(
            rows, split="validation", history=p["history"], horizons=p["horizons"], selection=None
        )

        def open_evaluation(selection):
            bindings()
            return load_dataset(
                rows,
                split="test",
                history=p["history"],
                horizons=p["horizons"],
                selection=selection,
            )

        result = execute_comparison(
            p,
            train,
            validation,
            open_evaluation,
            ROOT / "training",
            device=e["device"],
            protocol_sha256=sha(PROTOCOL),
            execution_sha256=sha(ROOT / "execution_manifest.json"),
            started_at=start,
            integrity_check=bindings,
        )
        write(
            ROOT / "closeout.json",
            dict(
                completed=True,
                result_path=str(ROOT / "training/result.json"),
                result_sha256=sha(ROOT / "training/result.json"),
                development_utility_gate=result["development_utility_gate"],
                P100_authorized=False,
                P960_authorized=False,
                stage2_ready=False,
            ),
        )
    except Exception:
        if not (ROOT / "RUN_FAILURE.json").exists():
            write(
                ROOT / "RUN_FAILURE.json",
                dict(
                    traceback=traceback.format_exc(),
                    gate="UNEVALUATED_INCOMPLETE",
                    automatic_retry=False,
                ),
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify_sources, "run": run}[args.action]()


if __name__ == "__main__":
    main()
