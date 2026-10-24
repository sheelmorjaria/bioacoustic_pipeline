"""Weighted RTPE estimators, separate from frozen implementation-v1 sources.

Population weights must be positive and sum to one. The shared data objective is
sum_i w_i mean_{observed cells in origin i} (prediction-target)^2.
Ridge adds alpha/D * sum of squared non-intercept coefficients, D output cells.
Thus full support gives per-output weighted SSE + alpha ||beta||². Partial
support uses effective row weights w_i*D/observed_count_i; no per-output
renormalisation or hidden alpha rescaling occurs. Inputs must already use the
same training-fitted scaler as the neural arm. Ridge does not refit one.
"""

import math

import torch

from .rclat_rtpe import _features


def _weights(weights, n):
    if (
        weights.ndim != 1
        or len(weights) != n
        or not weights.is_floating_point()
        or weights.requires_grad
        or not torch.isfinite(weights).all()
        or not (weights > 0).all()
    ):
        raise ValueError("Fixed finite positive origin weights required")
    w = weights.detach().to(device="cpu", dtype=torch.float64)
    if not math.isclose(w.sum().item(), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Population weights must sum to one; never silently renormalise alpha")
    return w


def _origin_errors(prediction, target, observed):
    if (
        prediction.ndim < 2
        or prediction.shape != target.shape
        or observed.shape != target.shape
        or observed.dtype != torch.bool
        or not prediction.is_floating_point()
        or not target.is_floating_point()
        or prediction.device != target.device
        or observed.device != target.device
    ):
        raise ValueError("Matching floating predictions/targets and boolean support required")
    mask = observed.flatten(1)
    if not mask.any(1).all():
        raise ValueError("Every origin requires an observed target cell")
    if not torch.isfinite(prediction[observed]).all() or not torch.isfinite(target[observed]).all():
        raise ValueError("Observed predictions and targets must be finite")
    errors = (torch.where(observed, prediction, 0) - torch.where(observed, target, 0)).square()
    return errors.flatten(1).sum(1) / mask.sum(1)


def weighted_mse(prediction, target, observed, weights):
    w = _weights(weights, len(prediction)).to(prediction)
    return (w * _origin_errors(prediction, target, observed)).sum()


def minibatch_mse(prediction, target, observed, population_weights, indices):
    """Unbiased loss for uniformly sampled origin indices (with replacement).

    Return N/B sum_b w[index_b]*loss_b, never a locally normalised weighted
    mean. Caller must actually sample uniformly and bind rows to these indices.
    Averaging complete equal-size disjoint batches also gives the full loss.
    """
    w = _weights(population_weights, len(population_weights))
    if (
        indices.ndim != 1
        or indices.dtype != torch.int64
        or len(indices) != len(prediction)
        or not len(indices)
        or (indices < 0).any()
        or (indices >= len(w)).any()
    ):
        raise ValueError("Valid global population indices matching batch rows required")
    selected = w[indices.detach().cpu()].to(prediction)
    return (len(w) / len(indices)) * (selected * _origin_errors(prediction, target, observed)).sum()


class WeightedTrainingScaler:
    """Equal weighted origins; equal observed history cells within each feature.

    A feature wholly missing in one origin contributes no support for that
    feature. Renormalise its moments over supported origins only. No target
    values or evaluation inputs enter fit. Transform never changes moments.
    """

    def fit(self, x, valid, weights):
        clean = _features(x, valid).detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 3:
            raise ValueError("B,T,F histories required")
        w = _weights(weights, len(x))
        v = valid.detach().cpu()
        counts = v.sum(1)
        mass = (w[:, None] * (counts > 0)).sum(0)
        if not (mass > 0).all():
            raise ValueError("Every feature needs observed training support")
        cell_weight = w[:, None, None] / counts[:, None, :].clamp_min(1)
        self.mean = (clean * cell_weight).sum((0, 1)) / mass
        variance = (torch.where(v, (clean - self.mean).square(), 0) * cell_weight).sum(
            (0, 1)
        ) / mass
        self.scale = variance.sqrt().clamp_min(1e-8)
        return self

    def transform(self, x, valid):
        _features(x, valid)
        if not hasattr(self, "mean") or x.shape[-1] != len(self.mean):
            raise ValueError("Fit scaler and preserve feature dimensions")
        clean = torch.where(valid, x, self.mean.to(x))
        return torch.where(valid, (clean - self.mean.to(x)) / self.scale.to(x), 0)


class WeightedHistoryRidge:
    """Float64 square-root row weighting and Cholesky, shared across target masks.

    The intercept is unpenalised. Exact constant columns are removed; this is
    algebraically equivalent with a free intercept and penalised coefficients.
    Missing inputs become zero plus explicit validity indicators. All output
    columns with identical support share one factorisation. No model selection.
    """

    def __init__(self, alpha=1.0):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Finite positive alpha required")
        self.alpha = alpha

    @staticmethod
    def _design(x, valid):
        clean = _features(x, valid).detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 3:
            raise ValueError("B,T,F histories required")
        return torch.cat([clean.flatten(1), valid.detach().cpu().double().flatten(1)], 1)

    def fit(self, x, valid, targets, observed, weights):
        _origin_errors(torch.zeros_like(targets), targets, observed)
        w = _weights(weights, len(x))
        if targets.ndim != 3 or len(targets) != len(x):
            raise ValueError("B,horizon,feature targets required")
        a = self._design(x, valid)
        self.history_shape = tuple(x.shape[1:])
        self.output_shape = tuple(targets.shape[1:])
        self.active_columns = torch.any(a != a[:1], dim=0)
        a = a[:, self.active_columns]
        y = targets.detach().to(device="cpu", dtype=torch.float64).flatten(1)
        mask = observed.detach().cpu().flatten(1)
        if not mask.any(0).all():
            raise ValueError("Every output requires observed training support")
        D = y.shape[1]
        effective = w * D / mask.sum(1)
        groups = {}
        for j in range(D):
            key = mask[:, j].numpy().tobytes()
            groups.setdefault(key, []).append(j)
        self.coefficients = torch.zeros((a.shape[1], D), dtype=torch.float64)
        self.intercept = torch.zeros(D, dtype=torch.float64)
        self.factorizations = 0
        self.maximum_normal_residual = 0.0
        for columns in groups.values():
            support = mask[:, columns[0]]
            q = effective[support]
            ax = a[support]
            ay = y[support][:, columns]
            mx = (q[:, None] * ax).sum(0) / q.sum()
            my = (q[:, None] * ay).sum(0) / q.sum()
            aw = (ax - mx) * q.sqrt()[:, None]
            yw = (ay - my) * q.sqrt()[:, None]
            if a.shape[1]:
                normal = aw.T @ aw
                normal.diagonal().add_(self.alpha)
                rhs = aw.T @ yw
                factor = torch.linalg.cholesky(normal)
                beta = torch.cholesky_solve(rhs, factor)
                residual = (normal @ beta - rhs).norm() / rhs.norm().clamp_min(1e-30)
                self.maximum_normal_residual = max(self.maximum_normal_residual, residual.item())
                self.factorizations += 1
            else:
                beta = torch.zeros((0, len(columns)), dtype=torch.float64)
            self.coefficients[:, columns] = beta
            self.intercept[columns] = my - mx @ beta
        return self

    def predict(self, x, valid):
        if not hasattr(self, "coefficients") or tuple(x.shape[1:]) != self.history_shape:
            raise ValueError("Fit model first and preserve history dimensions")
        a = self._design(x, valid)[:, self.active_columns]
        return (a @ self.coefficients + self.intercept).reshape(len(a), *self.output_shape).to(x)
