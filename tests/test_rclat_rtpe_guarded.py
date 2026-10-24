import ast
import inspect
import textwrap

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from bioacoustic_pipeline.representation.rclat_rtpe_pool import PoolTemporalPredictionEngine
from bioacoustic_pipeline.representation.rclat_rtpe_weighted import minibatch_mse
from bioacoustic_pipeline.representation.rclat_rtpe_guarded import (
    GuardedPoolTemporalPredictionEngine,
    guarded_features,
    guarded_minibatch_mse,
)


class NoBooleanIndex(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) == "aten.index.Tensor":
            assert not any(
                isinstance(index, torch.Tensor) and index.dtype == torch.bool
                for index in args[1]
                if index is not None
            )
        return func(*args, **(kwargs or {}))


def test_forward_arithmetic_differs_only_in_guard_call():
    original = ast.parse(textwrap.dedent(inspect.getsource(PoolTemporalPredictionEngine.forward)))
    candidate = ast.parse(
        textwrap.dedent(inspect.getsource(GuardedPoolTemporalPredictionEngine.forward))
    )

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "_features":
                node.id = "guarded_features"
            return node

    assert ast.dump(Rename().visit(original)) == ast.dump(candidate)


def test_cpu_outputs_losses_and_gradients_are_unchanged_with_masked_nonfinite():
    torch.manual_seed(101)
    original = PoolTemporalPredictionEngine(
        3, hidden=8, scales=(1, 4, 16), horizons=(4, 8, 16)
    ).double()
    candidate = GuardedPoolTemporalPredictionEngine(
        3, hidden=8, scales=(1, 4, 16), horizons=(4, 8, 16)
    ).double()
    candidate.load_state_dict(original.state_dict())
    x = torch.randn(4, 32, 3, dtype=torch.float64)
    v = torch.ones_like(x, dtype=torch.bool)
    v[:, 10, 2] = False
    x[~v] = float("nan")
    y = torch.randn(4, 3, 3, dtype=torch.float64)
    m = torch.ones_like(y, dtype=torch.bool)
    m[0, 1, 2] = False
    y[~m] = float("inf")
    w = torch.full((8,), 0.125, dtype=torch.float64)
    indices = torch.arange(4)

    def compute(model, fn):
        leaf = x.clone().requires_grad_()
        p = model(leaf, v)["forecast"][:, -1]
        loss = fn(p, y, m, w, indices)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        return p.detach(), loss.detach(), g, leaf.grad

    a = compute(original, minibatch_mse)
    with NoBooleanIndex():
        b = compute(candidate, guarded_minibatch_mse)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("position", [0, 12, 23])
def test_loss_guards_reject_observed_and_allow_unobserved_invalid_values(position):
    p = torch.ones(4, 3, 2)
    y = torch.zeros_like(p)
    m = torch.ones_like(p, dtype=torch.bool)
    w = torch.full((4,), 0.25, dtype=torch.float64)
    indices = torch.arange(4)
    y.flatten()[position] = float("nan")
    with NoBooleanIndex(), pytest.raises(ValueError, match="targets"):
        guarded_minibatch_mse(p, y, m, w, indices)
    m.flatten()[position] = False
    with NoBooleanIndex():
        assert torch.isfinite(guarded_minibatch_mse(p, y, m, w, indices))
    p.flatten()[position] = float("inf")
    m.flatten()[position] = True
    y.flatten()[position] = 0
    with NoBooleanIndex(), pytest.raises(ValueError, match="predictions"):
        guarded_minibatch_mse(p, y, m, w, indices)


def test_input_and_empty_origin_contracts_preserved():
    x = torch.ones(4, 3, 2)
    mask = torch.ones_like(x, dtype=torch.bool)
    x[-1, -1, -1] = float("inf")
    with pytest.raises(ValueError, match="features"):
        guarded_features(x, mask)
    mask[-1, -1, -1] = False
    assert torch.isfinite(guarded_features(x, mask)).all()
    w = torch.full((4,), 0.25, dtype=torch.float64)
    mask[0] = False
    with pytest.raises(ValueError, match="Every origin"):
        guarded_minibatch_mse(torch.ones_like(x), x, mask, w, torch.arange(4))


def test_full_population_loss_uses_same_weighted_objective():
    from bioacoustic_pipeline.representation.rclat_rtpe_guarded import guarded_weighted_mse
    from bioacoustic_pipeline.representation.rclat_rtpe_weighted import weighted_mse

    p = torch.arange(24, dtype=torch.float64).reshape(4, 3, 2) / 10
    y = torch.zeros_like(p)
    mask = torch.ones_like(p, dtype=torch.bool)
    mask[2, 1, 1] = False
    y[2, 1, 1] = float("nan")
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    expected = weighted_mse(p, y, mask, weights)
    with NoBooleanIndex():
        actual = guarded_weighted_mse(p, y, mask, weights)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
