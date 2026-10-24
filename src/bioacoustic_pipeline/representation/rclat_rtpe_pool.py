"""Equivalent causal pooled means, isolated from frozen cumulative-sum implementation."""

import torch
from torch.nn import functional as F

from .rclat_rtpe import TemporalPredictionEngine, _features


class PoolTemporalPredictionEngine(TemporalPredictionEngine):
    """Same parameters and mathematical causal means; fixed local pooling primitive.

    Avoids cumulative-sum/gather backward for missing-cell XPU inputs. This is
    a separately versioned numerical candidate, not a silent predecessor fix.
    """

    def forward(self, x, valid):
        clean = _features(x, valid)
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
