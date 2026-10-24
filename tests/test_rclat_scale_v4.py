"""Numerical invariants for the versioned composite-softmax execution path."""

import pytest
import torch

from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text
from bioacoustic_pipeline.representation.rclat_numerics import (
    check_gradients,
    check_parameters_and_optimizer,
    require_finite,
)
from bioacoustic_pipeline.representation.rclat_scale_v3 import ScaleCTC as Original
from bioacoustic_pipeline.representation.rclat_scale_v4 import ScaleCTC, stable_softmax


def test_composite_softmax_double_precision_gradient_check():
    torch.manual_seed(0)
    scores = torch.randn(2, 2, 4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(stable_softmax, (scores,), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(stable_softmax(scores), scores.softmax(-1))


def test_finite_padding_and_extreme_scores_remain_stable():
    scores = torch.tensor(
        [[1e4, 9999.0, -10000.0], [-10000.0, -10000.0, -10000.0]], requires_grad=True
    )
    output = stable_softmax(scores)
    output.square().sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(scores.grad).all()
    torch.testing.assert_close(output.sum(-1), torch.ones(2))
    torch.testing.assert_close(output, scores.softmax(-1))


def test_original_schema_initialization_forward_and_backward_are_preserved_on_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    original = Original(4, width=16, layers=2, heads=2, expansion=32, kernel=3)
    torch.manual_seed(0)
    corrected = ScaleCTC(4, width=16, layers=2, heads=2, expansion=32, kernel=3)
    for name, value in original.state_dict().items():
        assert torch.equal(value, corrected.state_dict()[name])
    x = torch.randn(2, 49, 4)
    x[0, 33:] = float("nan")
    mask = torch.ones_like(x, dtype=torch.bool)
    results = []
    for model in [original, corrected]:
        logits, lengths = model(x, mask, torch.tensor([33, 49]))
        loss = ctc_objective(logits, lengths, [encode_text("AA"), encode_text("THE")])
        loss.backward()
        check_gradients(model)
        results.append((logits.detach(), {n: p.grad for n, p in model.named_parameters()}))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-5, rtol=1e-4)
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], atol=1e-5, rtol=1e-4)


def test_guard_reports_nonfinite_gradient_name():
    model = torch.nn.Linear(2, 1)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model.weight.grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="weight"):
        check_gradients(model)


def test_optimizer_state_checked_even_when_parameters_are_finite():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    check_parameters_and_optimizer(model, optimizer)
    optimizer.state[model.weight]["exp_avg"].fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="exp_avg"):
        check_parameters_and_optimizer(model, optimizer)


def test_guard_does_not_hide_finite_values_with_overflowing_fp32_norm():
    value = torch.tensor([1e30, 1e30])
    assert require_finite([("large", value)], "test") == 1
    assert not torch.isfinite(torch.linalg.vector_norm(value))
