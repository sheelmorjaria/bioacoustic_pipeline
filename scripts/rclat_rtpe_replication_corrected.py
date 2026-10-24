"""Separately frozen five-seed RTPE development replication; never modifies RTPE-1."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import torch

from bioacoustic_pipeline.diagnostics.rclat_rtpe_execution import load_dataset, sha, write
from bioacoustic_pipeline.diagnostics.rclat_rtpe_replication import (
    seed_spec,
    replication_summary,
    unlock_replication,
)
from bioacoustic_pipeline.diagnostics.rclat_rtpe_replication_corrected import (
    train_seed,
    evaluate_seed,
    assert_global_lock,
)
from bioacoustic_pipeline.representation.rclat_rtpe_guarded import (
    GuardedPoolTemporalPredictionEngine as PoolTemporalPredictionEngine,
    guarded_minibatch_mse,
)

ROOT = Path("outputs/rclat_rtpe_multiseed_v2_execution")
BASE = Path("outputs/rclat_rtpe1_protocol_v2/protocol.json")
PAIRS = [
    (101, 2026102311),
    (211, 2026102312),
    (307, 2026102313),
    (401, 2026102314),
    (503, 2026102315),
]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bindings():
    p = json.loads((ROOT / "protocol.json").read_text())
    e = json.loads((ROOT / "execution_manifest.json").read_text())
    if sha(ROOT / "protocol.json") != e["protocol_sha256"]:
        raise ValueError("Replication protocol changed")
    for source, digest in e["dependencies"].items():
        if sha(source) != digest:
            raise ValueError("Execution dependency changed: " + source)
    if sha(BASE) != p["base_protocol_sha256"]:
        raise ValueError("Closed base protocol changed")
    base = json.loads(BASE.read_text())
    for mapping in [base["dependencies"], base["preflight_artifacts"]]:
        if not all(sha(source) == digest for source, digest in mapping.items()):
            raise ValueError("Closed dependency changed")
    if e["environment"] != dict(
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__
    ):
        raise ValueError("Environment version changed")
    for name in ["input_verification.json", "preflight.json"]:
        path = ROOT / name
        if path.exists() and json.loads(path.read_text())["execution_sha256"] != sha(
            ROOT / "execution_manifest.json"
        ):
            raise ValueError("Verified execution manifest changed")
    return p, base


def prepare():
    frozen = Path("outputs/rclat_rtpe_multiseed_v2_spec/protocol.json")
    preparation = json.loads(
        Path("outputs/rclat_rtpe_multiseed_v2_spec/preparation_manifest.json").read_text()
    )
    if sha(frozen) != preparation["protocol_sha256"]:
        raise ValueError("Corrective specification changed")
    for source, digest in preparation["dependencies"].items():
        if sha(source) != digest:
            raise ValueError("Preparation dependency changed: " + source)
    original = load_script("rtpe_original", "scripts/rclat_rtpe_train.py")
    original.bindings()
    ROOT.mkdir(parents=True, exist_ok=False)
    (ROOT / "protocol.json").write_bytes(frozen.read_bytes())
    paths = [
        Path(__file__),
        BASE,
        frozen,
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_replication_corrected.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_replication.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_execution.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_protocol.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_guarded.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_pool.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_weighted.py"),
        Path("tests/test_rclat_rtpe_corrected_execution.py"),
        Path("tests/test_rclat_rtpe_corrected_staging.py"),
        Path("outputs/rclat_rtpe_guard_execution_v1/certificate.json"),
        Path("outputs/rclat_rtpe1_protocol_v2/training_numerical_fixtures.npz"),
    ]
    write(
        ROOT / "execution_manifest.json",
        dict(
            status="PINNED_CORRECTED_RUNNER_NOT_AUTHORIZED_FOR_SCIENTIFIC_EXECUTION",
            protocol_sha256=sha(ROOT / "protocol.json"),
            dependencies={str(p): sha(p) for p in paths},
            environment=dict(
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__
            ),
            entrypoint="scripts/rclat_rtpe_replication_corrected.py execute",
            allowed_preparation_actions=["verify", "preflight"],
            authorization="Separate AUTHORIZATION.json binding protocol/execution hashes required;not created by preparation",
            scientific_training_started=False,
        ),
    )
    print("Corrected runner pinned;scientific execution NOT authorized.", flush=True)


def require_authorization():
    path = ROOT / "AUTHORIZATION.json"
    if not path.is_file():
        raise ValueError("Explicit scientific execution authorisation required")
    auth = json.loads(path.read_text())
    if not (
        auth.get("authorized") is True
        and auth.get("action") == "execute_five_seed_scientific_replication"
        and auth.get("protocol_sha256") == sha(ROOT / "protocol.json")
        and auth.get("execution_sha256") == sha(ROOT / "execution_manifest.json")
    ):
        raise ValueError("Pinned scientific execution authorisation required")


def run_grad(model, x, v, y, m, w, indices):
    model.zero_grad(set_to_none=True)
    leaf = x.detach().requires_grad_()
    prediction = model(leaf, v)["forecast"][:, -1]
    prediction.retain_grad()
    loss = guarded_minibatch_mse(prediction, y, m, w, indices)
    loss.backward()
    return [
        prediction.detach().cpu(),
        loss.detach().cpu().reshape(1),
        torch.cat(
            [p.grad.detach().cpu().flatten() for p in model.parameters() if p.grad is not None]
        ),
        leaf.grad.detach().cpu(),
        prediction.grad.detach().cpu(),
    ]


def verify():
    p, base = bindings()
    if (ROOT / "RUN_REQUESTED.json").exists():
        raise RuntimeError("Existing attempt preserved")
    rows = json.loads(Path(base["inventory"]).read_text())
    for i, r in enumerate(rows):
        if sha(r["source_path"]) != r["source_sha256"] or sha(r["cache_path"]) != r["cache_sha256"]:
            raise ValueError("Frozen source/cache changed")
        if (i + 1) % 500 == 0:
            print(f"Checked{i + 1}/2814source/cache pairs", flush=True)
    write(
        ROOT / "input_verification.json",
        dict(
            passed=True,
            execution_sha256=sha(ROOT / "execution_manifest.json"),
            records=len(rows),
            original_verified_inventory_sha256=sha(base["inventory"]),
            scientific_optimizer_updates=0,
            evaluation_scores_computed=0,
        ),
    )


def preflight():
    bindings()
    if not json.loads((ROOT / "input_verification.json").read_text())["passed"]:
        raise ValueError("Verified sources required")
    if (ROOT / "PREFLIGHT_STARTED.json").exists() or (ROOT / "RUN_REQUESTED.json").exists():
        raise FileExistsError("Existing preflight preserved;no retry")
    if (
        os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384"
        or "DNNL_PRIMITIVE_CACHE_CAPACITY" in os.environ
    ):
        raise ValueError("Frozen cache policy required")
    if not torch.xpu.is_available():
        raise RuntimeError("XPU0 unavailable")
    write(
        ROOT / "PREFLIGHT_STARTED.json",
        dict(execution_sha256=sha(ROOT / "execution_manifest.json"), optimizer_updates=0),
    )
    errors = []
    try:
        torch.set_num_threads(2)
        torch.xpu.set_device(0)
        with np.load(
            "outputs/rclat_rtpe1_protocol_v2/training_numerical_fixtures.npz", allow_pickle=False
        ) as f:
            data = {k: f[k] for k in f.files}
        for seed, _ in PAIRS:
            for case in range(3):
                directory = ROOT / f"preflight_seed_{seed}_case_{case}"
                directory.mkdir(exist_ok=False)
                write(directory / "STARTED.json", dict(seed=seed, case=case))
                torch.manual_seed(seed)
                cpu = PoolTemporalPredictionEngine(
                    64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16)
                )
                gpu = copy.deepcopy(cpu).to("xpu:0")
                x, v, y, m, w, indices = [
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
                a = run_grad(cpu, x, v, y, m, w, indices)
                b = run_grad(
                    gpu, x.to("xpu:0"), v.to("xpu:0"), y.to("xpu:0"), m.to("xpu:0"), w, indices
                )
                if not all(torch.isfinite(t).all() for t in a + b):
                    raise ValueError("Nonfinite numerical comparison")
                relative = [
                    float(
                        (left.double() - right.double()).norm()
                        / left.double().norm().clamp_min(1e-30)
                    )
                    for left, right in zip(a, b)
                ]
                record = dict(
                    seed=seed, case=case, relative_errors=relative, passed=max(relative) < 1e-4
                )
                torch.save(dict(CPU=a, XPU=b), directory / "parity_tensors.pt")
                write(directory / "result.json", record)
                if not record["passed"]:
                    raise ValueError("Registered numerical preflight failed")
                errors.append(record)
                del gpu
                torch.xpu.empty_cache()
        bindings()
        write(
            ROOT / "preflight.json",
            dict(
                passed=True,
                cases=errors,
                execution_sha256=sha(ROOT / "execution_manifest.json"),
                optimizer_updates=0,
            ),
        )
        print("All15corrected preflight cases passed;zero optimiser updates.", flush=True)
    except Exception:
        write(
            ROOT / "PREFLIGHT_FAILURE.json",
            dict(
                traceback=traceback.format_exc(),
                completed_cases=errors,
                automatic_retry=False,
                scientific_gate="UNEVALUATED",
            ),
        )
        raise


def execute():
    require_authorization()
    p, base = bindings()
    if (ROOT / "RUN_REQUESTED.json").exists() or (ROOT / "FAILURE.json").exists():
        raise RuntimeError("No retry/resume over existing study")
    if not json.loads((ROOT / "input_verification.json").read_text())["passed"]:
        raise ValueError("Verified sources required")
    active = subprocess.check_output(["ps", "-eo", "args="], text=True)
    if any(
        "rclat_bilstm_dropout_train.py" in line or "scripts/rclat_rtpe_train.py run" in line
        for line in active.splitlines()
    ):
        raise RuntimeError("XPU admission deferred for existing research execution")
    if (
        os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY") != "16384"
        or "DNNL_PRIMITIVE_CACHE_CAPACITY" in os.environ
    ):
        raise ValueError("Frozen cache policy required")
    if not torch.xpu.is_available():
        raise RuntimeError("XPU0 unavailable")
    try:
        if not json.loads((ROOT / "preflight.json").read_text())["passed"]:
            raise ValueError("Passed corrected preflight required")
        started = time.monotonic()
        write(
            ROOT / "RUN_REQUESTED.json",
            dict(
                protocol_sha256=sha(ROOT / "protocol.json"),
                execution_sha256=sha(ROOT / "execution_manifest.json"),
                started_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )

        def guard():
            bindings()
            if time.monotonic() - started > 36000:
                raise TimeoutError("Global study walltime")

        records = json.loads(Path(base["inventory"]).read_text())
        train = load_dataset(
            records, split="train", history=32, horizons=[4, 8, 16], selection=None
        )
        validation = load_dataset(
            records, split="validation", history=32, horizons=[4, 8, 16], selection=None
        )
        selections = {}
        trainings = {}
        for seed, sampler in PAIRS:
            guard()
            print(f"Training initialisation seed{seed},sampler{sampler}", flush=True)
            variant = seed_spec(base, initialization_seed=seed, sampler_seed=sampler)
            directory = ROOT / f"seed_{seed}"
            selection = train_seed(
                variant,
                train,
                validation,
                directory,
                device="xpu:0",
                protocol_sha256=sha(ROOT / "protocol.json"),
                execution_sha256=sha(ROOT / "execution_manifest.json"),
                integrity_check=guard,
            )
            selections[str(seed)] = selection
            trainings[str(seed)] = dict(
                path=str(directory),
                selection_sha256=sha(directory / "selection.json"),
                complete_sha256=sha(directory / "TRAINING_COMPLETE.json"),
            )
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        unlock_replication(selections, pairs=PAIRS, updates=5000, interval=250)
        write(
            ROOT / "global_selection_lock.json",
            dict(pairs=PAIRS, updates=5000, interval=250, trainings=trainings),
        )
        assert_global_lock(ROOT / "global_selection_lock.json")
        print(
            "All25000updates complete;global selection locked;opening new evaluation.", flush=True
        )
        evaluation = load_dataset(
            records,
            split="test",
            history=32,
            horizons=[4, 8, 16],
            selection=dict(completed_updates=25000),
        )
        results = {}
        baseline_prediction = None
        for seed, sampler in PAIRS:
            variant = seed_spec(base, initialization_seed=seed, sampler_seed=sampler)
            directory = ROOT / f"seed_{seed}"
            result = evaluate_seed(
                variant,
                directory,
                lambda: evaluation,
                ROOT / "global_selection_lock.json",
                device="xpu:0",
                integrity_check=guard,
            )
            results[str(seed)] = result
            with np.load(directory / "evaluation_forecasts.npz", allow_pickle=False) as f:
                prediction = f["ridge"]
            if baseline_prediction is None:
                baseline_prediction = prediction
            elif not np.array_equal(baseline_prediction, prediction):
                raise ValueError("Shared deterministic Ridge comparison did not reproduce exactly")
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        primary_index = base["horizons"].index(base["primary_horizon"])
        first = results[str(PAIRS[0][0])]
        ridge = {s: v[primary_index] for s, v in first["ridge"]["speakers"].items()}
        gru = {
            seed: {s: v[primary_index] for s, v in r["gru_selected"]["speakers"].items()}
            for seed, r in results.items()
        }
        summary = replication_summary(
            ridge, gru, pairs=PAIRS, bootstrap_seed=base["uncertainty"]["seed"], replicates=10000
        )
        summary.update(
            completed=True,
            total_GRU_updates=25000,
            baseline_forecasts_bit_identical=True,
            elapsed_seconds=time.monotonic() - started,
            selected_steps={s: r["selection"]["gru_step"] for s, r in results.items()},
        )
        write(ROOT / "result.json", summary)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in summary.items()
                    if k
                    not in [
                        "per_seed",
                        "seed_speaker_difference_matrix",
                        "per_speaker_mean_difference",
                    ]
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception:
        if not (ROOT / "FAILURE.json").exists():
            write(
                ROOT / "FAILURE.json",
                dict(
                    traceback=traceback.format_exc(),
                    gate="UNEVALUATED_INCOMPLETE",
                    automatic_retry=False,
                ),
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "preflight", "execute"])
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify, "preflight": preflight, "execute": execute}[
        args.action
    ]()


if __name__ == "__main__":
    main()
