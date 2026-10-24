"""Prepare, numerically check and freeze RTPE-1; no scientific model training."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch

from bioacoustic_pipeline.representation.rclat_rtpe import TemporalPredictionEngine
from bioacoustic_pipeline.representation.rclat_rtpe_weighted import (
    WeightedTrainingScaler,
    minibatch_mse,
)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def check(root):
    plan = json.loads((root / "candidate.json").read_text())
    for p, d in plan["dependencies"].items():
        if sha(p) != d:
            raise ValueError("Changed dependency: " + p)
    return plan


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    base = Path("outputs/rclat_rtpe_common_inventory_v1")
    paths = [
        base / "selected_origins.json",
        base / "sampling_policy.json",
        base / "summary.json",
        Path("outputs/rclat_rtpe_weighted_preflight_v1/summary.json"),
        Path(__file__),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_weighted.py"),
        Path("src/bioacoustic_pipeline/diagnostics/rclat_rtpe_protocol.py"),
        Path("tests/test_rclat_rtpe_protocol.py"),
    ]
    plan = dict(
        study="RCLAT RTPE-1 T0 weighted Ridge versus T1 causal GRU",
        status="CANDIDATE_CONDITIONAL_ON_PREFLIGHT_NOT_FROZEN",
        population="Frozen175/38/38speaker inventory;15840/3320/3352origins;previously exposed development corpus",
        inventory=str(base / "selected_origins.json"),
        history=32,
        horizons=[4, 8, 16],
        primary_horizon=8,
        frontend="Existing verified64D trailing mel;640samples160hop16kHz;no input/target waveform overlap",
        arms=[
            "Weighted history-matched Ridge",
            "Causal multiscale GRU64hidden;scales1,4,16;singleGRU;noorder/symbol/behaviour supervision",
        ],
        scaler="One weighted training-input-only scaler shared with both arms and all target horizons; origin weights fixed; no per-recording/evaluation adaptation",
        training_objective="Equal speaker/recording/origin weighted mean observed target-cell SSE across all3horizons; evaluation primary80ms only",
        ridge=dict(
            alpha_candidates=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            penalty="alpha/D * sum squared non-intercept coefficients; no per-output weight renormalisation",
            fit="CPUfloat64Cholesky;constantcolumnsremoved;identicalsupportsharesfactorisation",
            select="Minimum validation80ms speaker-macroMSE;earliest listed alpha on exact tie",
        ),
        gru=dict(
            hidden=64,
            seed=17,
            updates=5000,
            batch_origins=128,
            optimizer="AdamW",
            learning_rate=0.001,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0.0001,
            gradient_clip_norm=5.0,
            precision="FP32;noautocast",
            sampling="Uniform indices with replacement from frozen15840training origins;dedicatedCPUgeneratorseed2026102301",
            loss="N/B sum selected origin weights * per-origin observed-cell mean squared error;no batch-weight renormalisation",
            validation_every=250,
            checkpoint_every=250,
            selection="After all5000updates:minimumvalidation80msspeaker-macroMSE;earliestupdateexacttie;reportfinal5000",
            fresh_initialization=True,
            resume=False,
            early_stopping=False,
        ),
        evaluation=dict(
            primary="80ms MSE:observedcellsmeanwithinorigin,eightoriginsmeanwithinrecording,equalrecordingmeanwithinspeaker,equal38speakermean",
            contrast="Ridge loss minus GRU loss;positivefavoursGRU",
            secondary="40/160ms losses and finalGRU checkpoint descriptive only;cannot rescue primary",
            seal="No internalevaluation forward pass/scoring before both arms complete and validation selection is locked",
        ),
        uncertainty=dict(
            method="Paired whole-speaker percentile bootstrap;resample38speaker IDs uniformly with replacement;allrecordings/origins retained",
            seed=2026102302,
            replicates=10000,
            quantiles=[0.025, 0.975],
            quantile_method="linear",
            scope="Conditional on fitted single-seed models,fixedorigins,selection and exposeddevelopmentpopulation;notmultiseed/selection/withinrecordingsampling coverage",
        ),
        decision="Development predictive-utility PASS iff both arms complete all checks,point Ridge-minus-GRU>0 and lower pairedspeaker95percentile bound>0. Zero/equality is not pass. No automatic ASR/P100/Stage2progression.",
        numerical_guards="Finite inputs/validtargetcells,logits,loss,allgradients,clipnorm,parameters,optimizerstates;recordexactstepandoriginIDs;no sanitisation",
        failure="No automaticretry,resume,budgetextension,LR/batch/precision/capacity change or silentrepair;incomplete execution leaves gate unevaluated; preserveartifacts",
        runtime=dict(
            device="Single XPU0 forGRU;CPU Ridge;XPU1 preflight replication only",
            requested_ONEDNN_PRIMITIVE_CACHE_CAPACITY=16384,
            max_walltime_seconds=7200,
            peak_host_RSS_GiB=8,
            post_fit_ridge_residual_max=1e-8,
            changes_to_running_experiments="prohibited",
        ),
        preflight=dict(
            CPU_condition_number_limit=1e8,
            regularised_matrix="Minimumgridalpha added to weighted centred inputGram",
            XPU_relative_tolerance=1e-4,
            metrics=["forecast", "loss", "global_parameter_gradients", "input_gradients"],
            real_training_fixtures=3,
            devices=[0, 1],
            optimizer_updates=0,
        ),
        runner="Scientific runner must be implemented/tested against this spec and pinned in a separate execution manifest before start;noautomaticstart",
        final_test_access=False,
        P100_authorized=False,
        P960_authorized=False,
        stage2_ready=False,
        dependencies={str(p): sha(p) for p in paths},
    )
    write(root / "candidate.json", plan)
    started = time.monotonic()
    torch.set_num_threads(2)
    records = [
        r for r in json.loads((base / "selected_origins.json").read_text()) if r["split"] == "train"
    ]
    n = sum(len(r["selected_origins"]) for r in records)
    x = torch.empty(n, 32, 64)
    valid = torch.ones_like(x, dtype=torch.bool)
    weights = []
    fixture_rows = [0, len(records) // 2, len(records) - 1]
    fixtures = []
    offset = 0
    for number, r in enumerate(records):
        if sha(r["cache_path"]) != r["cache_sha256"]:
            raise ValueError("Training cache changed: " + r["uid"])
        with np.load(r["cache_path"], allow_pickle=False) as f:
            mel = f["mel"]
            mask = f["mel_valid"]
        for origin in r["selected_origins"]:
            x[offset] = torch.from_numpy(mel[origin - 31 : origin + 1])
            valid[offset] = torch.from_numpy(mask[origin - 31 : origin + 1])
            weights.append(r["origin_weight_within_split"])
            offset += 1
        if number in fixture_rows:
            chosen = r["selected_origins"][:4]
            fixtures.append(
                dict(
                    uid=r["uid"],
                    indices=np.arange(offset - 8, offset - 4),
                    x=np.stack([mel[t - 31 : t + 1] for t in chosen]),
                    valid=np.stack([mask[t - 31 : t + 1] for t in chosen]),
                    y=mel[np.array(chosen)[:, None] + np.array([4, 8, 16])[None]],
                    observed=mask[np.array(chosen)[:, None] + np.array([4, 8, 16])[None]],
                )
            )
    w = torch.tensor(weights, dtype=torch.float64)
    scaler = WeightedTrainingScaler().fit(x, valid, w)
    z = scaler.transform(x, valid)
    a = z.double().flatten(1)
    mean = (w[:, None] * a).sum(0)
    a = (a - mean) * w.sqrt()[:, None]
    gram = a.T @ a
    eig = torch.linalg.eigvalsh(gram)
    alpha = min(plan["ridge"]["alpha_candidates"])
    condition = float((eig[-1] + alpha) / (eig[0].clamp_min(0) + alpha))
    if not torch.isfinite(eig).all() or condition > 1e8:
        raise ValueError("Real-input regularised conditioning gate failed")
    gram.diagonal().add_(alpha)
    factor = torch.linalg.cholesky(gram)
    if not torch.isfinite(factor).all():
        raise ValueError("Real-input factorisation nonfinite")
    arrays = {"population_weights": w.numpy()}
    for i, f in enumerate(fixtures):
        arrays[f"x_{i}"] = scaler.transform(
            torch.from_numpy(f["x"]), torch.from_numpy(f["valid"])
        ).numpy()
        arrays[f"valid_{i}"] = f["valid"]
        arrays[f"y_{i}"] = scaler.transform(
            torch.from_numpy(f["y"]), torch.from_numpy(f["observed"])
        ).numpy()
        arrays[f"observed_{i}"] = f["observed"]
        arrays[f"indices_{i}"] = f["indices"]
    # One fixture also verifies missing-cell arithmetic; does not alter source masks.
    arrays["valid_2"] = arrays["valid_2"].copy()
    arrays["valid_2"][:, 10, 2] = False
    arrays["x_2"][:, 10, 2] = np.nan
    arrays["observed_2"] = arrays["observed_2"].copy()
    arrays["observed_2"][0, 0, 2] = False
    arrays["y_2"][0, 0, 2] = np.nan
    np.savez_compressed(root / "training_numerical_fixtures.npz", **arrays)
    write(
        root / "scaler_preflight.json",
        dict(
            mean=scaler.mean.tolist(),
            scale=scaler.scale.tolist(),
            origin="Weighted training inputs only;noevaluationcacheaccess;no predictive model fitted",
        ),
    )
    write(
        root / "cpu_conditioning.json",
        dict(
            passed=True,
            training_origins=n,
            training_cache_entries_verified=len(records),
            input_columns=2048,
            weighted_gram_smallest_eigenvalue=float(eig[0]),
            weighted_gram_largest_eigenvalue=float(eig[-1]),
            minimum_alpha=alpha,
            regularised_condition_number=condition,
            cholesky_finite=True,
            elapsed_seconds=time.monotonic() - started,
            peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            real_predictive_models_fitted=0,
            scalers_fitted=1,
            optimizer_updates=0,
            evaluation_cache_entries_read=0,
            fixture_uids=[f["uid"] for f in fixtures],
        ),
    )
    print(
        "Real training-input conditioning passed; no target-based model fitting or held-out scoring.",
        flush=True,
    )


def run_grad(model, x, valid, y, observed, w, indices):
    model.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_()
    forecast = model(x, valid)["forecast"][:, -1]
    loss = minibatch_mse(forecast, y, observed, w, indices)
    loss.backward()
    grads = torch.cat(
        [p.grad.detach().cpu().flatten() for p in model.parameters() if p.grad is not None]
    )
    if not torch.isfinite(grads).all():
        raise ValueError("Nonfinite arithmetic")
    return [forecast.detach().cpu(), loss.detach().cpu().reshape(1), grads, x.grad.detach().cpu()]


def xpu_check(root):
    plan = check(root)
    torch.set_num_threads(2)
    if not torch.xpu.is_available() or torch.xpu.device_count() < 2:
        raise RuntimeError("Both registered XPUs must be accessible")
    with np.load(root / "training_numerical_fixtures.npz", allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    receipt = dict(
        candidate_sha256=sha(root / "candidate.json"),
        fixture_sha256=sha(root / "training_numerical_fixtures.npz"),
    )
    write(root / "XPU_STARTED.json", receipt)
    rows = []
    for device in plan["preflight"]["devices"]:
        for case in range(3):
            torch.manual_seed(17)
            model = TemporalPredictionEngine(64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16))
            gpu = copy.deepcopy(model).to(f"xpu:{device}")
            x = torch.from_numpy(data[f"x_{case}"])
            v = torch.from_numpy(data[f"valid_{case}"])
            y = torch.from_numpy(data[f"y_{case}"])
            m = torch.from_numpy(data[f"observed_{case}"])
            w = torch.from_numpy(data["population_weights"])
            indices = torch.from_numpy(data[f"indices_{case}"])
            cpu = run_grad(model, x, v, y, m, w, indices)
            xp = run_grad(
                gpu,
                x.to(f"xpu:{device}"),
                v.to(f"xpu:{device}"),
                y.to(f"xpu:{device}"),
                m.to(f"xpu:{device}"),
                w,
                indices,
            )
            errors = {
                name: float((a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-30))
                for name, a, b in zip(plan["preflight"]["metrics"], cpu, xp)
            }
            if any(not np.isfinite(e) or e >= 1e-4 for e in errors.values()):
                raise ValueError("XPU parity threshold failed: " + str(errors))
            rows.append(dict(device=device, case=case, relative_errors=errors))
            print(json.dumps(rows[-1]), flush=True)
            del gpu
            torch.xpu.empty_cache()
    write(root / "xpu_parity.json", dict(passed=True, cases=rows, optimizer_updates=0, **receipt))


def freeze(root):
    plan = check(root)
    cpu = json.loads((root / "cpu_conditioning.json").read_text())
    xpu = json.loads((root / "xpu_parity.json").read_text())
    if not cpu["passed"] or not xpu["passed"] or len(xpu["cases"]) != 6:
        raise ValueError("All prospective prerequisites must pass")
    if xpu["candidate_sha256"] != sha(root / "candidate.json") or xpu["fixture_sha256"] != sha(
        root / "training_numerical_fixtures.npz"
    ):
        raise ValueError("Preflight artifact binding failed")
    final = {
        **plan,
        "status": "LOCALLY_FROZEN_SCIENTIFIC_SPECIFICATION_NOT_EXTERNAL_REGISTRATION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_artifacts": {
            str(root / name): sha(root / name)
            for name in [
                "cpu_conditioning.json",
                "xpu_parity.json",
                "training_numerical_fixtures.npz",
                "scaler_preflight.json",
            ]
        },
        "scientific_training_started": False,
        "scientific_runner_implemented": False,
        "automatic_start": False,
    }
    write(root / "protocol.json", final)
    lines = [
        "# RTPE-1 scientific specification freeze",
        "",
        "The T0 weighted history-matched Ridge versus T1 causal multiscale GRU scientific specification is locally frozen. All six registered CPU–XPU comparisons passed. No scientific training, optimiser updates or internal-evaluation scoring occurred. A scientific runner and separately pinned execution manifest are still required before start.",
        "",
        f"Real training-input regularised condition number: {cpu['regularised_condition_number']:.6g} at alpha={cpu['minimum_alpha']}. Weighted Gram Cholesky passed; no real-target Ridge fit occurred.",
        f"Maximum registered CPU–XPU relative error: {max(e for row in xpu['cases'] for e in row['relative_errors'].values()):.6g}. Three fixed training fixtures crossed both devices; one includes explicitly synthetic missing cells.",
        "",
        "Ridge alpha grid: 0.0001,0.001,0.01,0.1,1,10. GRU: hidden64,scales1/4/16,seed17,5000updates,batch128,AdamW LR0.001/weightdecay0.0001,clip5,FP32. Every250updates validate and checkpoint; complete5000updates before selecting minimum80ms validation loss, earliest exact tie. No adaptive budget extension or retries.",
        "",
        "Primary: internal-evaluation speaker-macro80ms standardised MSE; equal recordings within speaker and eight frozen origins within recording; paired Ridge-minus-GRU contrast. Development utility requires positive point and strictly positive lower95percentile speaker-bootstrap bound (10000draws,seed2026102302). All38evaluation speakers remain sealed until fits and validation-only selections are locked.",
        "",
        "Bootstrap interpretation is conditional on fixed single-seed selected models and fixed origin sample. It is not certified coverage for model fitting, selection, source selection or within-recording origin sampling. This previously exposed development population is not independent confirmation. No communication crossed bootstrap is reused. Secondary horizons/final checkpoint cannot rescue a failed primary result.",
        "",
        "Source inventory, sampler, estimator and predecessor artifacts remain unchanged. Temporal-order, linguistic and behavioural objectives are inactive. P100/P960 and Stage2 remain blocked. See protocol.json for exact budgets, failure rules and dependencies.",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (root / "README.md").write_text(
        "Scientific specification locally frozen; scientific runner not yet implemented; no scientific training started. See RESULTS.md and protocol.json.\n"
    )
    write(root / "manifest.json", {str(p): sha(p) for p in root.iterdir() if p.is_file()})
    print("RTPE-1 specification frozen; no automatic scientific execution.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "xpu", "freeze"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        {"prepare": prepare, "xpu": xpu_check, "freeze": freeze}[args.action](args.output)
    except Exception:
        if args.output.exists():
            failure = args.output / (args.action + "_FAILURE.json")
            if not failure.exists():
                write(
                    failure,
                    dict(
                        traceback=traceback.format_exc(),
                        optimizer_updates=0,
                        scientific_training_started=False,
                    ),
                )
        raise


if __name__ == "__main__":
    main()
