"""Bounded-history causal acoustic forecasting; no implicit semantic supervision.

Call each model on independently constructed windows. No recording concatenation,
feature extraction, label alignment, scientific fitting or split selection occurs here.
"""

import math
import re

import torch
from torch import nn
from torch.nn import functional as F


def _features(x, valid):
    if (
        x.ndim not in (2, 3)
        or x.shape != valid.shape
        or valid.dtype != torch.bool
        or not x.is_floating_point()
        or x.device != valid.device
        or not all(x.shape)
    ):
        raise ValueError("Matching nonempty floating features and boolean validity required")
    if not torch.isfinite(x[valid]).all():
        raise ValueError("Observed features must be finite")
    return torch.where(valid, x, torch.zeros_like(x))


def _positive_tuple(values, name):
    values = tuple(values)
    if (
        not values
        or any(type(v) is not int or v <= 0 for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain unique positive integers")
    return values


class TemporalPredictionEngine(nn.Module):
    """Unidirectional GRU over current values plus causal masked moving means.

    Scales and horizons are frame counts, not linguistic boundaries. Forecast
    outputs at t use only frames <=t. Caller freezes frame cadence and history.
    Optional symbols are unnormalised logits, not inferred phonetic references.
    No behavioural head is activated by constructing this model.
    """

    def __init__(self, features, *, hidden=64, scales=(1, 4, 16), horizons=(1, 4), symbols=None):
        super().__init__()
        if type(features) is not int or features <= 0 or type(hidden) is not int or hidden <= 0:
            raise ValueError("Positive feature and hidden dimensions required")
        self.features = features
        self.scales = _positive_tuple(scales, "Scales")
        self.horizons = _positive_tuple(horizons, "Horizons")
        self.encoder = nn.GRU(features * 2 * len(self.scales), hidden, batch_first=True)
        self.forecast = nn.Linear(hidden, len(self.horizons) * features)
        self.order = nn.Linear(hidden, 1)
        if symbols is not None and (type(symbols) is not int or symbols < 2):
            raise ValueError("Symbol inventory must include at least two classes")
        self.symbols = nn.Linear(hidden, symbols) if symbols is not None else None

    def forward(self, x, valid):
        clean = _features(x, valid)
        if x.ndim != 3 or x.shape[-1] != self.features or not valid.any(-1).all():
            raise ValueError("Use supported, unpadded B,T,F history windows")
        count = valid.to(x.dtype)
        sums = F.pad(clean.cumsum(1), (0, 0, 1, 0))
        counts = F.pad(count.cumsum(1), (0, 0, 1, 0))
        ends = torch.arange(1, x.shape[1] + 1, device=x.device)
        inputs = []
        for scale in self.scales:
            starts = (ends - scale).clamp_min(0)
            n = counts[:, ends] - counts[:, starts]
            mean = (sums[:, ends] - sums[:, starts]) / n.clamp_min(1)
            inputs.extend([mean, (n > 0).to(x.dtype)])
        states, _ = self.encoder(torch.cat(inputs, -1))
        result = {
            "states": states,
            "forecast": self.forecast(states).reshape(
                x.shape[0], x.shape[1], len(self.horizons), self.features
            ),
            "order_logits": self.order(states[:, -1]).squeeze(-1),
        }
        if self.symbols is not None:
            result["symbols"] = self.symbols(states)
        return result


def forecast_windows(x, valid, *, history, horizons):
    """One recording only. Targets remain actual future observations, never imputed.

    A wholly unsupported frame anywhere from history start to furthest target
    excludes the origin. Partial feature gaps remain explicit masks. Forecast
    losses evaluate only actually observed target cells on common support.
    """
    _features(x, valid)
    horizons = _positive_tuple(horizons, "Horizons")
    if x.ndim != 2 or type(history) is not int or history <= 0:
        raise ValueError("One T,F recording and positive history required")
    support = valid.any(-1)
    origins = [
        t
        for t in range(history - 1, len(x) - max(horizons))
        if bool(support[t - history + 1 : t + max(horizons) + 1].all())
    ]
    if not origins:
        raise ValueError("No eligible contiguous history/target windows")
    origin = torch.tensor(origins, device=x.device)
    index = origin[:, None] + torch.tensor(horizons, device=x.device)[None]
    return dict(
        inputs=torch.stack([x[t - history + 1 : t + 1] for t in origins]),
        input_valid=torch.stack([valid[t - history + 1 : t + 1] for t in origins]),
        targets=x[index],
        target_valid=valid[index],
        origins=origin,
    )


def masked_mse(prediction, target, observed):
    if (
        prediction.shape != target.shape
        or observed.shape != target.shape
        or observed.dtype != torch.bool
        or not observed.any()
    ):
        raise ValueError(
            "Matching predictions, targets and nonempty boolean target support required"
        )
    if not torch.isfinite(target[observed]).all() or not torch.isfinite(prediction[observed]).all():
        raise ValueError("Observed predictions and targets must be finite")
    return (prediction[observed] - target[observed]).square().mean()


class HistoryRidge:
    """Train-fitted standardisation and per-target Ridge on flattened values + masks.

    Exactly the same input windows and target cells as the neural arm. Alpha is
    supplied by the caller; no evaluation-driven tuning occurs in this class.
    """

    def __init__(self, alpha=1.0):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Positive finite Ridge alpha required")
        self.alpha = alpha

    def _design(self, x, valid, fit=False):
        clean = _features(x, valid).detach().to(device="cpu", dtype=torch.float64)
        if x.ndim != 3:
            raise ValueError("B,T,F history windows required")
        shape = tuple(x.shape[1:])
        v = valid.detach().cpu()
        if fit:
            self.history_shape = shape
            n = v.sum((0, 1))
            if not (n > 0).all():
                raise ValueError("Every input feature needs training support")
            self.mean = clean.sum((0, 1)) / n
            self.scale = torch.sqrt(
                torch.where(v, (clean - self.mean).square(), 0).sum((0, 1)) / n
            ).clamp_min(1e-8)
        elif not hasattr(self, "weights") or shape != self.history_shape:
            raise ValueError("Fit Ridge first and preserve history shape")
        z = torch.where(v, (clean - self.mean) / self.scale, 0)
        a = torch.cat([z.flatten(1), v.to(z.dtype).flatten(1)], 1)
        return torch.cat([torch.ones(len(a), 1, dtype=a.dtype), a], 1)

    def fit(self, x, valid, targets, observed):
        masked_mse(torch.zeros_like(targets), targets, observed)
        if targets.ndim != 3 or len(targets) != len(x):
            raise ValueError("Targets must be B,horizon,feature")
        a = self._design(x, valid, fit=True)
        y = targets.detach().cpu().double().flatten(1)
        m = observed.detach().cpu().flatten(1)
        self.output_shape = tuple(targets.shape[1:])
        penalty = torch.eye(a.shape[1], dtype=a.dtype) * self.alpha
        penalty[0, 0] = 0
        weights = []
        for j in range(y.shape[1]):
            if not m[:, j].any():
                raise ValueError("Each target requires training observations")
            aj = a[m[:, j]]
            weights.append(torch.linalg.solve(aj.T @ aj + penalty, aj.T @ y[m[:, j], j]))
        self.weights = torch.stack(weights, 1)
        return self

    def predict(self, x, valid):
        a = self._design(x, valid)
        return (a @ self.weights).reshape(len(a), *self.output_shape).to(x)


def permute_history(x, valid, permutation):
    """Move whole frames and masks together; never imply artifact matching.

    Caller supplies a frozen permutation. This preserves constituent feature
    observations, not boundary continuity or natural acoustic plausibility.
    """
    _features(x, valid)
    if (
        x.ndim != 3
        or permutation.dtype != torch.int64
        or permutation.shape != (x.shape[1],)
        or sorted(permutation.tolist()) != list(range(x.shape[1]))
    ):
        raise ValueError("A bijective history permutation is required")
    return x[:, permutation], valid[:, permutation]


def order_loss(logits, authentic, *, control_audit):
    """Auxiliary supervision disabled without an explicit independent control audit."""
    audit = control_audit or {}
    if audit.get("artifact_matching_approved") is not True or not re.fullmatch(
        "[a-fA-F0-9]{64}", str(audit.get("audit_sha256", ""))
    ):
        raise ValueError("Temporal-order loss requires an approved artifact-control audit")
    if (
        logits.shape != authentic.shape
        or not torch.isfinite(logits).all()
        or not ((authentic == 0) | (authentic == 1)).all()
    ):
        raise ValueError("Finite logits and binary authenticity targets required")
    return F.binary_cross_entropy_with_logits(logits, authentic.to(logits))


def validate_group_split(records):
    if not records:
        raise ValueError("Nonempty recording metadata required")
    owners = {"recording": {}, "individual": {}}
    for row in records:
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Explicit train/validation/test partition required")
        for field, mapping in owners.items():
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("Nonempty recording and individual identifiers required")
            if value in mapping and mapping[value] != row["split"]:
                raise ValueError(f"{field} crosses partitions")
            mapping[value] = row["split"]


class BehavioralReadout(nn.Module):
    """Inactive until external behavioural reference and use/transfer gates pass.

    Approval metadata is a software prerequisite, not independent certification.
    Features and context must contain observed history only. Training protocol
    and target matching remain external responsibilities.
    """

    def __init__(self, hidden, outcomes, *, reference):
        super().__init__()
        if (
            reference.get("independent_observation") is not True
            or reference.get("source_use_approved") is not True
            or reference.get("transfer_approved") is not True
            or not re.fullmatch("[a-fA-F0-9]{64}", str(reference.get("reference_sha256", "")))
        ):
            raise ValueError("Independent approved behavioural reference and transfer required")
        if type(hidden) is not int or hidden <= 0 or type(outcomes) is not int or outcomes < 2:
            raise ValueError("Positive hidden size and at least two outcomes required")
        self.head = nn.Linear(hidden, outcomes)

    def forward(self, states):
        return self.head(states)


class TrainingScaler:
    """Caller fits once on training frames; transform never updates statistics."""

    def fit(self, x, valid):
        clean = _features(x, valid).detach()
        axes = tuple(range(x.ndim - 1))
        n = valid.sum(axes)
        if not (n > 0).all():
            raise ValueError("Every feature requires observed training values")
        self.mean = clean.sum(axes) / n
        self.scale = torch.sqrt(
            torch.where(valid, (clean - self.mean).square(), 0).sum(axes) / n
        ).clamp_min(1e-8)
        return self

    def transform(self, x, valid):
        _features(x, valid)
        if not hasattr(self, "mean") or x.shape[-1] != len(self.mean):
            raise ValueError("Fit training scaler first and preserve feature dimensions")
        clean = torch.where(valid, x, self.mean.to(x))
        return torch.where(valid, (clean - self.mean.to(x)) / self.scale.to(x), 0)
