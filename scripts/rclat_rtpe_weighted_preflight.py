"""Synthetic weighted-estimator numerical/resource preflight; no acoustic fitting."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
import traceback

import numpy as np
from scipy.linalg import lstsq
import torch

from bioacoustic_pipeline.representation.rclat_rtpe import TemporalPredictionEngine
from bioacoustic_pipeline.representation.rclat_rtpe_weighted import (
    WeightedHistoryRidge,
    WeightedTrainingScaler,
    weighted_mse,
    minibatch_mse,
)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)


def checks():
    torch.manual_seed(51)
    x = torch.randn(12, 4, 3, dtype=torch.float64)
    v = torch.ones_like(x, dtype=torch.bool)
    v[2, 1, 0] = False
    x[~v] = float("nan")
    y = torch.randn(12, 2, 3, dtype=torch.float64)
    m = torch.ones_like(y, dtype=torch.bool)
    m[0, 0, 1] = False
    m[2, 1, 0] = False
    y[~m] = float("nan")
    w = torch.arange(1, 13, dtype=torch.float64)
    w /= w.sum()
    model = WeightedHistoryRidge(0.13).fit(x, v, y, m, w)
    a = np.column_stack([np.ones(12), np.where(v, x, 0).reshape(12, -1), v.numpy().reshape(12, -1)])
    penalty = np.eye(a.shape[1]) * np.sqrt(0.13)
    penalty[0, 0] = 0
    beta = []
    for j in range(6):
        observed = m.reshape(12, -1)[:, j].numpy()
        q = w.numpy() * 6 / m.reshape(12, -1).sum(1).numpy()
        lhs = np.vstack([a[observed] * np.sqrt(q[observed, None]), penalty])
        rhs = np.r_[
            y.reshape(12, -1).numpy()[observed, j] * np.sqrt(q[observed]), np.zeros(a.shape[1])
        ]
        beta.append(lstsq(lhs, rhs, lapack_driver="gelsd")[0])
    reference = a @ np.stack(beta, 1)
    error = np.linalg.norm(
        model.predict(x, v).numpy().reshape(12, -1) - reference
    ) / np.linalg.norm(reference)
    assert error < 1e-10
    neural = TemporalPredictionEngine(3, hidden=8, scales=(1, 4), horizons=(1, 2)).double()
    x = torch.randn(9, 8, 3, dtype=torch.float64)
    v = torch.ones_like(x, dtype=torch.bool)
    y = torch.randn(9, 2, 3, dtype=torch.float64)
    m = torch.ones_like(y, dtype=torch.bool)
    m[0, 0, 1] = False
    y[~m] = float("nan")
    w = torch.arange(1, 10, dtype=torch.float64)
    w /= w.sum()
    pred = neural(x, v)["forecast"][:, -1]
    params = list(neural.parameters())
    full = weighted_mse(pred, y, m, w)
    manual = sum(w[i] * (pred[i][m[i]] - y[i][m[i]]).square().mean() for i in range(9))

    def grads(loss, retain=False):
        values = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain)
        return torch.cat([a.flatten() for a in values if a is not None])

    full_grad = grads(full, True)
    manual_grad = grads(manual)
    total = 0
    for first in (0, 3, 6):
        indices = torch.arange(first, first + 3)
        p = neural(x[indices], v[indices])["forecast"][:, -1]
        total = total + minibatch_mse(p, y[indices], m[indices], w, indices) / 3
    batched_grad = grads(total)
    e1 = ((full_grad - manual_grad).norm() / full_grad.norm()).item()
    e2 = ((full_grad - batched_grad).norm() / full_grad.norm()).item()
    assert max(e1, e2) < 1e-10
    return dict(
        independent_svd_prediction_relative_error=float(error),
        small_ridge_normal_residual=model.maximum_normal_residual,
        manual_neural_gradient_relative_error=e1,
        minibatch_neural_gradient_relative_error=e2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = Path("outputs/rclat_rtpe_common_inventory_v1/selected_origins.json")
    paths = [
        inventory,
        Path(__file__),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe_weighted.py"),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("tests/test_rclat_rtpe_weighted.py"),
    ]
    plan = dict(
        status="FROZEN_SYNTHETIC_ENGINEERING_CHECK_NOT_SCIENTIFIC_PROTOCOL",
        seed=51,
        dependencies={str(p): sha(p) for p in paths},
        objective="sum_i w_i mean_observed_cells_i squared_error; positive population weights sum1",
        ridge_penalty="alpha/D * sum squared non-intercept coefficients; per-target solve effective weights w_i*D/observed_count_i; no per-target renormalisation",
        ridge_alpha=0.1,
        feature_scaling="One weighted training-input-only scaler shared by Ridge and GRU",
        minibatching="Uniform origins with replacement; estimator N/B sum w_i*loss_i; no local weight normalisation",
        numerical_relative_tolerance=1e-10,
        full_shape_normal_residual_tolerance=1e-9,
        resource_limit_seconds=180,
        peak_RSS_limit_GiB=4,
        synthetic_shape=[15840, 32, 64],
        target_shape=[15840, 3, 64],
        gru_batch=64,
        synthetic_only=True,
        real_data_models_fitted=0,
        scientific_freeze=False,
        no_real_feature_or_waveform_reads=True,
        CPU_only=True,
    )
    write(args.output / "plan.json", plan)
    write(args.output / "STARTED.json", dict(plan_sha256=sha(args.output / "plan.json")))

    def timeout_handler(*_):
        raise TimeoutError("Frozen engineering walltime reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        numerical = checks()
        print("Independent numerical and GRU gradient checks passed.", flush=True)
        # Only identities/indices/weights are read; no real acoustic targets.
        rows = [r for r in json.loads(inventory.read_text()) if r["split"] == "train"]
        w = torch.tensor(
            [r["origin_weight_within_split"] for r in rows for _ in r["selected_origins"]],
            dtype=torch.float64,
        )
        n = len(w)
        assert n == 15840
        torch.manual_seed(plan["seed"])
        raw = torch.randn(n, 32, 64)
        valid = torch.ones_like(raw, dtype=torch.bool)
        targets = torch.randn(n, 3, 64)
        observed = torch.ones_like(targets, dtype=torch.bool)
        scaler = WeightedTrainingScaler().fit(raw, valid, w)
        x = scaler.transform(raw, valid)
        del raw
        gc.collect()
        fit_started = time.monotonic()
        model = WeightedHistoryRidge(0.1).fit(x, valid, targets, observed, w)
        fit_seconds = time.monotonic() - fit_started
        assert model.maximum_normal_residual < 1e-9 and model.factorizations == 1
        prediction = model.predict(x[:128], valid[:128])
        assert torch.isfinite(prediction).all()
        neural = TemporalPredictionEngine(64, hidden=64, scales=(1, 4, 16), horizons=(4, 8, 16))
        indices = torch.arange(64)
        forecast = neural(x[indices], valid[indices])["forecast"][:, -1]
        loss = minibatch_mse(forecast, targets[indices], observed[indices], w, indices)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in neural.parameters() if p.grad is not None)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        elapsed = time.monotonic() - started
        assert peak <= 4 and elapsed < 180
        assert all(sha(p) == digest for p, digest in plan["dependencies"].items())
        signal.alarm(0)
        result = dict(
            completed=True,
            numerical=numerical,
            full_shape_normal_residual=model.maximum_normal_residual,
            full_shape_factorizations=model.factorizations,
            active_design_columns=int(model.active_columns.sum()),
            ridge_fit_seconds=fit_seconds,
            elapsed_seconds=elapsed,
            peak_RSS_GiB=peak,
            synthetic_ridge_fits=1,
            neural_backward_checks=1,
            optimizer_updates=0,
            real_data_models_fitted=0,
            source="Synthetic Gaussian arrays with frozen real-inventory dimensions and weights only",
            CPU_preflight_passed=True,
            XPU_preflight_performed=False,
            scientific_execution_ready=False,
            P100_authorized=False,
            P960_authorized=False,
            stage2_ready=False,
        )
        write(args.output / "summary.json", result)
        lines = [
            "# RTPE-1 weighted-estimator engineering verification",
            "",
            "Weighted numerical checks and full-shape CPU resource preflight passed. No real acoustic target was fitted and no optimiser update occurred.",
            "",
            f"Independent augmented-SVD prediction relative error: {numerical['independent_svd_prediction_relative_error']:.3g}.",
            f"Manual neural gradient relative error: {numerical['manual_neural_gradient_relative_error']:.3g}; minibatch vs full population gradient error: {numerical['minibatch_neural_gradient_relative_error']:.3g}.",
            f"Full synthetic shape: 15,840 x 32 x 64 inputs and 15,840 x 3 x 64 targets, using exact frozen training origin weights. Peak process RSS: {peak:.3f} GiB; total elapsed: {elapsed:.2f}s; Ridge fit: {fit_seconds:.2f}s.",
            f"Normal-equation residual: {model.maximum_normal_residual:.3g}. Shared target support used one factorisation; exact constant mask columns were removed, retaining {int(model.active_columns.sum())} columns.",
            "",
            "Convention: weights are strictly positive and sum to one; data loss is weighted per-origin observed-cell mean squared error. Ridge adds alpha/D times total squared non-intercept coefficients. Its per-target row weights are w_i D / observed_count_i without target-specific renormalisation. A free intercept is not penalised. This fixes the alpha convention; final scientific alpha candidates still require a new freeze.",
            "",
            "Both arms consume the same externally weighted-scaled windows. The weighted scaler uses equal observed history cells within each origin/feature and the frozen origin weights across supported origins. Evaluation never updates scaler moments. Uniform-origin minibatches use N/B times the weighted batch sum, yielding an unbiased full-objective gradient; locally normalising batch weights would change that objective.",
            "",
            "Independent reference checks used square-root row weighting and an augmented SVD solve, separate from the Cholesky production solver. Tests include partial missing targets, all-constant designs, invalid weights and exact speaker/recording/origin aggregation. FP32 neural losses retain FP32 arithmetic; Ridge and the reference solve use CPU float64.",
            "",
            "Scope: CPU synthetic engineering validation only. This does not certify XPU arithmetic, real-data conditioning, scientific predictive utility, stochastic training convergence, final batch size/budgets, checkpoint selection or paired uncertainty. No real waveform/cache was read, no scientific model trained, and no model weights saved for scientific reuse. Frozen predecessor sources and the running dropout experiment remain unchanged. RTPE-1 remains unfrozen; P100/P960 and Stage 2 remain blocked.",
        ]
        (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
        (args.output / "README.md").write_text(
            "See RESULTS.md and summary.json. Weighted-estimator CPU engineering checks passed; scientific protocol remains unfrozen.\n"
        )
        write(
            args.output / "manifest.json",
            {str(p): sha(p) for p in args.output.iterdir() if p.is_file()},
        )
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        signal.alarm(0)
        write(
            args.output / "FAILURE.json",
            dict(traceback=traceback.format_exc(), real_data_models_fitted=0),
        )
        raise


if __name__ == "__main__":
    main()
