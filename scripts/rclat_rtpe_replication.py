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
from bioacoustic_pipeline.diagnostics.rclat_rtpe_replication_execution import (
    train_seed,
    evaluate_seed,
    assert_global_lock,
)
from bioacoustic_pipeline.representation.rclat_rtpe_pool import PoolTemporalPredictionEngine

ROOT = Path("outputs/rclat_rtpe_multiseed_v1")
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
    original = load_script("rtpe_original", "scripts/rclat_rtpe_train.py")
    base, _ = original.bindings()
    assert json.loads((original.ROOT / "scientific_closeout.json").read_text())["verified"]
    ROOT.mkdir(parents=True, exist_ok=False)
    paths = [
        BASE,
        Path("outputs/rclat_rtpe1_execution_v1/scientific_closeout.json"),
        Path("outputs/rclat_rtpe_common_inventory_v1/selected_origins.json"),
        Path(__file__),
        Path("scripts/rclat_rtpe_train.py"),
        Path("scripts/rclat_rtpe_protocol_prepare_v2.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_execution.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_replication.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_replication_execution.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_protocol.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_pool.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_weighted.py"),
        Path("tests/test_rclat_rtpe_replication.py"),
        Path("tests/test_rclat_rtpe_replication_execution.py"),
    ]
    protocol = dict(
        study="RTPE-R1 finite-five-seed development replication",
        status="LOCALLY_FROZEN_NOT_EXTERNAL_REGISTRATION",
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        base_protocol=str(BASE),
        base_protocol_sha256=sha(BASE),
        seed_pairs=[dict(initialization=a, sampler=b) for a, b in PAIRS],
        seed_order=[a for a, _ in PAIRS],
        original_seed17="Historical excluded from new replication gate",
        changes="Only initialisation and minibatch RNG streams differ; fixed data/model/loss/optimizer/budgets/validation selection retained. New staging seals evaluation until all5fits finish.",
        model_budget_per_seed=5000,
        total_GRU_updates=25000,
        updates_before_any_new_evaluation=25000,
        device="XPU0 sequential for all5seeds;CPU Ridge;no device confounding",
        ridge="Refit full6alpha grid per pipeline to preserve tested workflow;require bit-identical evaluation forecasts;analyse as one shared deterministic comparator,not5independent Ridge replicates",
        primary="For each seed:38speaker-macro80ms Ridge-minus-GRU contrast and original10000draw pairedspeaker95percentile interval",
        replication_criterion="PASS iff all5complete and all5individually pass original positive point and strictly positive lower95bound. Any complete seed failure gives finite5gateFAIL. Incomplete run gives UNEVALUATED;no dropped/replaced seeds.",
        aggregate="Report equal-seed mean of paired speaker-macro contrasts,all5conditional intervals,between-seed macro sampleSD/range,per-speaker seed means and descriptive seed-by-speaker residual RMS",
        uncertainty="No joint seed/speaker interval,seed-population confidence claim or biological-replicate multiplication. Original speaker bootstrap is conditional per seed;seed spread descriptive.",
        reporting="Report all5selected checkpoints,final checkpoints,alpha selections,allseed/speaker contrasts;no best-seed selection/ensembling or secondary rescue",
        evaluation_seal="Global lock requires every 5000update ledger and20validation selections,checkpoint/scaler hashes;only then load internal-evaluation features for prediction",
        runtime=dict(
            per_seed_active_seconds=7200,
            total_study_wall_seconds=36000,
            host_RSS_GiB=8,
            active_time="Each seed training+evaluation,excluding waiting while other seeds fit;global walltime includes all stages",
            cache_capacity=16384,
        ),
        numerical_preflight="3fixed real-training fixtures x5new initialisation seeds onXPU0;FP32 CPU-XPU relative tolerance1e-4;zerooptimizerupdates",
        failure="No retry/resume,seed replacement,budget extension,altered sampler/LR/precision/capacity,or outcome-driven tuning;preserve all failures",
        new_source_transfer=False,
        development_exposure="Same already exposed population and original source/sample partitions;not independent confirmation",
        automatic_start=False,
        scientific_training_started=False,
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
    )
    write(ROOT / "protocol.json", protocol)
    write(
        ROOT / "execution_manifest.json",
        dict(
            protocol_sha256=sha(ROOT / "protocol.json"),
            dependencies={str(f): sha(f) for f in paths},
            environment=dict(
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__
            ),
            base_execution_staged="Original algorithm copied into new module;only sampler-seed parameterisation and deferred evaluation changed",
            entrypoint="scripts/rclat_rtpe_replication.py execute",
            scientific_training_started=False,
        ),
    )
    print(
        "Five-seed protocol and staged execution manifest locally frozen; no training.", flush=True
    )


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
    p, base = bindings()
    helper = load_script("numerics", "scripts/rclat_rtpe_protocol_prepare_v2.py")
    with np.load(
        "outputs/rclat_rtpe1_protocol_v2/training_numerical_fixtures.npz", allow_pickle=False
    ) as f:
        data = {k: f[k] for k in f.files}
    errors = []
    torch.set_num_threads(2)
    for seed, _ in PAIRS:
        for case in range(3):
            torch.manual_seed(seed)
            cpu = PoolTemporalPredictionEngine(
                64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16)
            )
            gpu = copy.deepcopy(cpu).to("xpu:0")
            x = torch.from_numpy(data[f"x_{case}"])
            v = torch.from_numpy(data[f"valid_{case}"])
            y = torch.from_numpy(data[f"y_{case}"])
            m = torch.from_numpy(data[f"observed_{case}"])
            w = torch.from_numpy(data["population_weights"])
            indices = torch.from_numpy(data[f"indices_{case}"])
            a = helper.run_grad(cpu, x, v, y, m, w, indices)
            b = helper.run_grad(
                gpu, x.to("xpu:0"), v.to("xpu:0"), y.to("xpu:0"), m.to("xpu:0"), w, indices
            )
            error = max(
                float(
                    (left.double() - right.double()).norm() / left.double().norm().clamp_min(1e-30)
                )
                for left, right in zip(a, b)
            )
            if not np.isfinite(error) or error >= 1e-4:
                raise ValueError("Registered numerical preflight failed")
            errors.append(dict(seed=seed, case=case, maximum_relative_error=error))
            del gpu
            torch.xpu.empty_cache()
    write(
        ROOT / "preflight.json",
        dict(
            passed=True,
            cases=errors,
            execution_sha256=sha(ROOT / "execution_manifest.json"),
            optimizer_updates=0,
        ),
    )
    print("All15CPU-XPU numerical checks passed; zero optimiser updates.", flush=True)


def execute():
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
        preflight()
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
    parser.add_argument("action", choices=["prepare", "verify", "execute"])
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify, "execute": execute}[args.action]()


if __name__ == "__main__":
    main()
