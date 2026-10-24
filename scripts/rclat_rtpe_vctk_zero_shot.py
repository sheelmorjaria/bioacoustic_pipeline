"""Pinned zero-shot preparation/inference;never fits or selects a model."""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback

import numpy as np
import scipy
import soundfile as sf
import torch
from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import sha, write
from bioacoustic_pipeline.diagnostics.rclat_vctk_transfer_data import (
    build_source_arrays,
    load_frozen,
    equal_states,
)
from bioacoustic_pipeline.diagnostics.rclat_vctk_transfer_execution import run_inference_only
from bioacoustic_pipeline.representation.rclat_rtpe_guarded import guarded_minibatch_mse
from bioacoustic_pipeline.representation.rclat_causal_resample_v2 import causal_decimate_v2

ROOT = Path("outputs/rclat_rtpe_vctk_zero_shot_execution_v1")
SPEC = Path("outputs/rclat_rtpe_vctk_transfer_protocol_v1")
ARCHIVE = Path("data/rclat_vctk/VCTK-Corpus-0.92.zip")
FIXTURE = Path("outputs/rclat_rtpe1_protocol_v2/training_numerical_fixtures.npz")


def environment():
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        soundfile=sf.__version__,
        libsndfile=sf.__libsndfile_version__,
    )


def prepare():
    p = json.loads((SPEC / "protocol.json").read_text())
    assert all(sha(q) == h for q, h in p["dependencies"].items())
    ROOT.mkdir(exist_ok=False)
    paths = [
        Path(__file__),
        SPEC / "protocol.json",
        SPEC / "selected_inventory.json",
        SPEC / "independent_sampling_audit.json",
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_transfer_data.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_vctk_transfer_execution.py"),
        Path("tests/test_rclat_vctk_transfer_data.py"),
        Path("tests/test_rclat_vctk_transfer_execution.py"),
        Path("tests/test_rclat_vctk_zero_shot_runner.py"),
        FIXTURE,
    ]
    write(
        ROOT / "execution_manifest.json",
        dict(
            status="PINNED_INFERENCE_RUNNER_NOT_AUTHORIZED",
            protocol_sha256=sha(SPEC / "protocol.json"),
            inventory_sha256=sha(SPEC / "selected_inventory.json"),
            dependencies={str(q): sha(q) for q in paths},
            environment=environment(),
            actions=["verify", "preflight", "execute"],
            model_fitting=False,
            model_selection=False,
            scientific_execution_authorized=False,
        ),
    )
    print("Inference-only execution manifest pinned;no predictions or scores.")


def bindings():
    p = json.loads((SPEC / "protocol.json").read_text())
    e = json.loads((ROOT / "execution_manifest.json").read_text())
    if (
        e["protocol_sha256"] != sha(SPEC / "protocol.json")
        or e["inventory_sha256"] != sha(SPEC / "selected_inventory.json")
        or e["environment"] != environment()
    ):
        raise ValueError("Frozen protocol/inventory/environment changed")
    if not all(sha(q) == h for q, h in e["dependencies"].items()) or not all(
        sha(q) == h for q, h in p["dependencies"].items()
    ):
        raise ValueError("Pinned scientific dependency changed")
    for info in p["models"]:
        for item in info["files"].values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Frozen model bytes changed")
    return p, e


def runtime_settings(gpu=False):
    if os.environ.get("OPENBLAS_NUM_THREADS") != "2" or os.environ.get("OMP_NUM_THREADS") != "2":
        raise ValueError("Frozen CPU thread settings required")
    torch.set_num_threads(2)
    if gpu:
        if (
            os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384"
            or "DNNL_PRIMITIVE_CACHE_CAPACITY" in os.environ
        ):
            raise ValueError("Frozen XPU cache policy required")
        if not torch.xpu.is_available():
            raise RuntimeError("XPU0 unavailable")
        torch.xpu.set_device(0)


def require_authorization():
    path = ROOT / "SCIENTIFIC_AUTHORIZATION.json"
    if not path.is_file():
        raise ValueError("Explicit scientific inference authorisation required")
    a = json.loads(path.read_text())
    if not (
        a.get("authorized") is True
        and a.get("protocol_sha256") == sha(SPEC / "protocol.json")
        and a.get("execution_sha256") == sha(ROOT / "execution_manifest.json")
        and a.get("source_verification_sha256") == sha(ROOT / "source_verification.json")
        and a.get("preflight_sha256") == sha(ROOT / "preflight.json")
    ):
        raise ValueError("Hash-bound scientific inference authorisation required")


def verify():
    p, e = bindings()
    runtime_settings()
    stage = ROOT / "verification"
    stage.mkdir(exist_ok=False)
    write(stage / "STARTED.json", dict(model_inference=False, predictive_scores=False))
    try:
        started = time.monotonic()

        def limits():
            if (
                time.monotonic() - started > 7200
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20 > 8
            ):
                raise RuntimeError("Frozen preparation resource bound")

        receipt = json.loads(
            Path(
                "outputs/rclat_rtpe_vctk_support_execution_v2/support/archive_receipt.json"
            ).read_text()
        )
        if sha(ARCHIVE) != receipt["sha256"]:
            raise ValueError("Verified source archive changed")
        with np.load(p["preprocessing"]["coefficients"], allow_pickle=False) as f:
            if not np.array_equal(f["taps"], causal_decimate_v2(np.zeros(600), 48000)["taps"]):
                raise ValueError("Certified resampler coefficients changed")
        reference = None
        for info in p["models"]:
            model, ridge, scaler, state = load_frozen(info)
            if reference is None:
                reference = state
            elif not equal_states(reference, state):
                raise ValueError("Frozen Ridge/scaler numerical states differ")
            del model, ridge, scaler
        inventory = json.loads((SPEC / "selected_inventory.json").read_text())

        def progress(number, total):
            if number % 128 == 0 or number == total:
                print(
                    f"Source/feature verification{number}/{total};no model predictions.", flush=True
                )

        data, ids = build_source_arrays(ARCHIVE, inventory, check=limits, progress=progress)
        if len(ids) != 27648:
            raise ValueError("Protected population size changed")
        with (ROOT / "source_arrays.npz").open("xb") as f:
            np.savez_compressed(f, **data)
        write(ROOT / "origin_ids.json", ids)
        bindings()
        limits()
        write(
            ROOT / "source_verification.json",
            dict(
                passed=True,
                records=3456,
                origins=len(ids),
                speakers=len({r["speaker"] for r in ids}),
                source_arrays_sha256=sha(ROOT / "source_arrays.npz"),
                origin_ids_sha256=sha(ROOT / "origin_ids.json"),
                execution_sha256=sha(ROOT / "execution_manifest.json"),
                frozen_Ridge_scaler_states_identical=True,
                model_inference=False,
                predictive_scores=False,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
    except Exception:
        write(stage / "FAILURE.json", dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise


def gradients(model, x, v, y, m, w, indices):
    model.zero_grad(set_to_none=True)
    leaf = x.detach().requires_grad_()
    forecast = model(leaf, v)["forecast"][:, -1]
    forecast.retain_grad()
    loss = guarded_minibatch_mse(forecast, y, m, w, indices)
    loss.backward()
    return [
        forecast.detach().cpu(),
        loss.detach().cpu().reshape(1),
        torch.cat(
            [t.grad.detach().cpu().flatten() for t in model.parameters() if t.grad is not None]
        ),
        leaf.grad.detach().cpu(),
        forecast.grad.detach().cpu(),
    ]


def preflight():
    p, e = bindings()
    runtime_settings(gpu=True)
    if not json.loads((ROOT / "source_verification.json").read_text())["passed"]:
        raise ValueError("Verified source cache required")
    stage = ROOT / "numerical_preflight"
    stage.mkdir(exist_ok=False)
    write(stage / "STARTED.json", dict(optimizer_updates=0, VCTK_model_predictions=False))
    try:
        with np.load(FIXTURE, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        records = []
        for info in p["models"]:
            cpu, _, _, _ = load_frozen(info)
            gpu = copy.deepcopy(cpu).to("xpu:0")
            for case in range(3):
                directory = stage / f"seed_{info['seed']}_case_{case}"
                directory.mkdir(exist_ok=False)
                write(directory / "STARTED.json", dict(seed=info["seed"], fixture=case))
                x, v, y, m, w, idx = [
                    torch.from_numpy(data[k])
                    for k in [
                        f"x_{case}",
                        f"valid_{case}",
                        f"y_{case}",
                        f"observed_{case}",
                        "population_weights",
                        f"indices_{case}",
                    ]
                ]
                a = gradients(cpu, x, v, y, m, w, idx)
                b = gradients(
                    gpu, x.to("xpu:0"), v.to("xpu:0"), y.to("xpu:0"), m.to("xpu:0"), w, idx
                )
                if not all(torch.isfinite(t).all() for t in a + b):
                    raise ValueError("Nonfinite trained-state numerical preflight")
                errors = [
                    float(
                        (left.double() - right.double()).norm()
                        / left.double().norm().clamp_min(1e-30)
                    )
                    for left, right in zip(a, b)
                ]
                record = dict(
                    seed=info["seed"], fixture=case, errors=errors, passed=max(errors) < 1e-4
                )
                torch.save(dict(CPU=a, XPU=b), directory / "tensors.pt")
                write(directory / "result.json", record)
                if not record["passed"]:
                    raise ValueError("Frozen trained-state parity tolerance failed")
                records.append(record)
            del gpu, cpu
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        bindings()
        write(
            ROOT / "preflight.json",
            dict(
                passed=True,
                cases=records,
                execution_sha256=sha(ROOT / "execution_manifest.json"),
                optimizer_updates=0,
                VCTK_model_predictions=False,
            ),
        )
    except Exception:
        write(stage / "FAILURE.json", dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
    print("All15trained-state numerical fixtures passed;no VCTK predictions.", flush=True)


def execute():
    require_authorization()
    p, e = bindings()
    runtime_settings(gpu=True)
    verification = json.loads((ROOT / "source_verification.json").read_text())
    preflight_record = json.loads((ROOT / "preflight.json").read_text())
    if (
        not verification["passed"]
        or not preflight_record["passed"]
        or verification["execution_sha256"] != sha(ROOT / "execution_manifest.json")
        or preflight_record["execution_sha256"] != sha(ROOT / "execution_manifest.json")
    ):
        raise ValueError("Verified execution prerequisites required")
    if (
        sha(ROOT / "source_arrays.npz") != verification["source_arrays_sha256"]
        or sha(ROOT / "origin_ids.json") != verification["origin_ids_sha256"]
    ):
        raise ValueError("Verified protected arrays changed")
    # Persist attempt before loading any external arrays or forecasting models.
    write(
        ROOT / "RUN_REQUESTED.json",
        dict(
            protocol_sha256=sha(SPEC / "protocol.json"),
            execution_sha256=sha(ROOT / "execution_manifest.json"),
            authorized_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    started = time.monotonic()

    def limits():
        if time.monotonic() - started > 7200:
            raise TimeoutError("Frozen inference walltime")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20 > 8:
            raise MemoryError("Frozen host RSS")

    try:
        with np.load(ROOT / "source_arrays.npz", allow_pickle=False) as f:
            raw = {k: torch.from_numpy(f[k]) for k in f.files}
        ids = json.loads((ROOT / "origin_ids.json").read_text())
        infos = {info["seed"]: info for info in p["models"]}

        def predict(seed):
            bindings()
            limits()
            model, ridge, scaler, _ = load_frozen(infos[seed], device="xpu:0")
            x = scaler.transform(raw["x"], raw["v"])
            y = scaler.transform(raw["y"], raw["m"])
            neural = []
            linear = []
            with torch.no_grad():
                for first in range(0, len(ids), 128):
                    limits()
                    last = first + 128
                    neural.append(
                        model(x[first:last].to("xpu:0"), raw["v"][first:last].to("xpu:0"))[
                            "forecast"
                        ][:, -1]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                for first in range(0, len(ids), 512):
                    limits()
                    last = first + 512
                    # Frozen predict performs CPU float64 design/coefficients and
                    # preserves its original output cast to FP32 input dtype.
                    linear.append(
                        ridge.predict(x[first:last], raw["v"][first:last]).detach().cpu().numpy()
                    )
            result = dict(
                ridge=np.concatenate(linear),
                gru=np.concatenate(neural),
                targets=y.numpy(),
                observed=raw["m"].numpy(),
            )
            del model, ridge, scaler
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
            bindings()
            return result

        result = run_inference_only(
            ROOT / "scientific_run",
            [info["seed"] for info in p["models"]],
            ids,
            predict,
            bootstrap_seed=p["uncertainty"]["seed"],
            replicates=p["uncertainty"]["replicates"],
            check=limits,
            context=dict(
                protocol_sha256=sha(SPEC / "protocol.json"),
                execution_sha256=sha(ROOT / "execution_manifest.json"),
                source_arrays_sha256=verification["source_arrays_sha256"],
            ),
        )
        bindings()
        limits()
        write(
            ROOT / "scientific_closeout.json",
            dict(
                completed=True,
                all_five_transfer_gate=result["all_five_transfer_gate"],
                elapsed_seconds=time.monotonic() - started,
                model_fitting=False,
                optimizer_updates=0,
            ),
        )
    except Exception:
        write(
            ROOT / "RUN_FAILURE.json",
            dict(
                traceback=traceback.format_exc(),
                gate="UNEVALUATED_INCOMPLETE",
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "preflight", "execute"])
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify, "preflight": preflight, "execute": execute}[
        args.action
    ]()
