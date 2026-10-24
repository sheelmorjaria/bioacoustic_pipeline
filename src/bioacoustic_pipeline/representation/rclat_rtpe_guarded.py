"""Guard-only candidate integration; frozen scientific entrypoints remain unchanged.

WHERE masks unsupported cells before finiteness reduction. Only validation
operators change; model parameters, pooled means, weighted errors, reductions,
weight normalisation and minibatch scaling retain their frozen expressions.
"""

import torch
from torch.nn import functional as F

from .rclat_rtpe_pool import PoolTemporalPredictionEngine
from .rclat_rtpe_weighted import _weights


def guarded_features(x, valid):
    if (
        x.ndim not in (2, 3)
        or x.shape != valid.shape
        or valid.dtype != torch.bool
        or not x.is_floating_point()
        or x.device != valid.device
        or not all(x.shape)
    ):
        raise ValueError("Matching nonempty floating features and boolean validity required")
    clean = torch.where(valid, x, torch.zeros_like(x))
    if not torch.isfinite(clean).all():
        raise ValueError("Observed features must be finite")
    return clean


def _guarded_origin_errors(prediction, target, observed):
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
    if not torch.isfinite(torch.where(observed, prediction, 0)).all():
        raise ValueError("Observed predictions must be finite")
    if not torch.isfinite(torch.where(observed, target, 0)).all():
        raise ValueError("Observed targets must be finite")
    errors = (torch.where(observed, prediction, 0) - torch.where(observed, target, 0)).square()
    return errors.flatten(1).sum(1) / mask.sum(1)


def guarded_weighted_mse(prediction, target, observed, weights):
    w = _weights(weights, len(prediction)).to(prediction)
    return (w * _guarded_origin_errors(prediction, target, observed)).sum()


def guarded_minibatch_mse(prediction, target, observed, population_weights, indices):
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
    return (len(w) / len(indices)) * (
        selected * _guarded_origin_errors(prediction, target, observed)
    ).sum()


class GuardedPoolTemporalPredictionEngine(PoolTemporalPredictionEngine):
    """Identical forward arithmetic/state; only input finiteness guard is replaced."""

    def forward(self, x, valid):
        clean = guarded_features(x, valid)
        if x.ndim != 3 or x.shape[-1] != self.features or not valid.any(-1).all():
            raise ValueError("Use supported, unpadded B,T,F history windows")
        inputs = []
        for scale in self.scales:
            total = (
                F.avg_pool1d(F.pad(clean.transpose(1, 2), (scale - 1, 0)), scale, stride=1) * scale
            )
            count = (
                F.avg_pool1d(
                    F.pad(valid.to(x.dtype).transpose(1, 2), (scale - 1, 0)), scale, stride=1
                )
                * scale
            )
            denominator = torch.where(count > 0, count, torch.ones_like(count))
            inputs.extend(
                [(total / denominator).transpose(1, 2), (count > 0).to(x.dtype).transpose(1, 2)]
            )
        states, _ = self.encoder(torch.cat(inputs, -1))
        result = dict(
            states=states,
            forecast=self.forecast(states).reshape(
                x.shape[0], x.shape[1], len(self.horizons), self.features
            ),
            order_logits=self.order(states[:, -1]).squeeze(-1),
        )
        if self.symbols is not None:
            result["symbols"] = self.symbols(states)
        return result
