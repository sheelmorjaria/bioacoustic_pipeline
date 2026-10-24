"""Independent weighted least-squares and neural reduction contracts."""

import numpy as np
import pytest
import torch
from scipy.linalg import lstsq

from bioacoustic_pipeline.representation.rclat_rtpe_weighted import (
    WeightedHistoryRidge,
    WeightedTrainingScaler,
    weighted_mse,
    minibatch_mse,
)


def fixture():
    torch.manual_seed(23)
    x = torch.randn(9, 3, 2, dtype=torch.float64)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[2, 1, 0] = False
    x[~valid] = float("nan")
    y = torch.randn(9, 2, 2, dtype=torch.float64)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 0, 1] = False
    mask[2, 1, 0] = False
    y[~mask] = float("nan")
    w = torch.arange(1, 10, dtype=torch.float64)
    w /= w.sum()
    return x, valid, y, mask, w


def test_ridge_matches_independent_augmented_svd_and_stationarity():
    x, v, y, m, w = fixture()
    alpha = 0.13
    model = WeightedHistoryRidge(alpha).fit(x, v, y, m, w)
    a = np.concatenate(
        [np.where(v.numpy(), x.numpy(), 0).reshape(9, -1), v.numpy().reshape(9, -1)], axis=1
    )
    a = np.column_stack([np.ones(9), a])
    D = 4
    penalty = np.sqrt(alpha) * np.eye(a.shape[1])
    penalty[0, 0] = 0
    coefficients = []
    for j in range(D):
        obs = m.numpy().reshape(9, -1)[:, j]
        weights = w.numpy() * D / m.numpy().reshape(9, -1).sum(1)
        lhs = np.vstack([a[obs] * np.sqrt(weights[obs, None]), penalty])
        rhs = np.r_[y.numpy().reshape(9, -1)[obs, j] * np.sqrt(weights[obs]), np.zeros(a.shape[1])]
        coefficients.append(lstsq(lhs, rhs, lapack_driver="gelsd")[0])
    reference = np.stack(coefficients, axis=1)
    np.testing.assert_allclose(
        model.predict(x, v).numpy().reshape(9, -1), a @ reference, rtol=1e-10, atol=1e-10
    )
    assert model.maximum_normal_residual < 1e-10
    assert model.factorizations == 3


def test_full_and_uniform_minibatch_gradients_match_manual_population_objective():
    _, _, y, m, w = fixture()
    p = torch.zeros_like(y, requires_grad=True)
    loss = weighted_mse(p, y, m, w)
    manual = sum(w[i] * (p[i][m[i]] - y[i][m[i]]).square().mean() for i in range(9))
    torch.testing.assert_close(loss, manual)
    grad = torch.autograd.grad(loss, p)[0]
    chunked = 0
    for start in range(0, 9, 3):
        indices = torch.arange(start, start + 3)
        chunked += minibatch_mse(p[indices], y[indices], m[indices], w, indices) / 3
    torch.testing.assert_close(torch.autograd.grad(chunked, p)[0], grad)
    assert torch.isfinite(grad).all() and not grad[~m].any()


def test_equal_speaker_recording_origin_loss_is_not_uniform_origin_loss():
    p = torch.tensor([[[1.0]], [[1.0]], [[3.0]]])
    y = torch.zeros_like(p)
    m = torch.ones_like(p, dtype=torch.bool)
    # Speaker A has two recordings, speaker B one; each has one origin here.
    w = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)
    assert weighted_mse(p, y, m, w).item() == pytest.approx(5.0)
    assert weighted_mse(p, y, m, w).item() != pytest.approx(11 / 3)


def test_scaler_uses_origin_weights_and_never_updates_at_evaluation():
    x = torch.tensor([[[0.0], [2.0]], [[8.0], [float("nan")]]], dtype=torch.float64)
    m = torch.isfinite(x)
    w = torch.tensor([0.75, 0.25], dtype=torch.float64)
    s = WeightedTrainingScaler().fit(x, m, w)
    assert s.mean.item() == pytest.approx(2.75)
    # Average within an origin before weighted averaging across origins.
    assert s.scale.square().item() == pytest.approx(
        0.75 * ((0 - 2.75) ** 2 + (2 - 2.75) ** 2) / 2 + 0.25 * (8 - 2.75) ** 2
    )
    assert torch.isfinite(s.transform(x, m)).all()
    mean = s.mean.clone()
    s.transform(torch.full_like(x, 100), torch.ones_like(m))
    assert torch.equal(mean, s.mean)


def test_invalid_weights_masks_and_shapes_are_rejected():
    x, v, y, m, w = fixture()
    for bad in [w * 2, -w, w * float("nan"), w[:-1]]:
        with pytest.raises(ValueError):
            WeightedHistoryRidge().fit(x, v, y, m, bad)
    with pytest.raises(ValueError):
        weighted_mse(torch.zeros_like(y), y, torch.zeros_like(m), w)
    model = WeightedHistoryRidge().fit(x, v, y, m, w)
    with pytest.raises(ValueError):
        model.predict(x[:, :2], v[:, :2])
    with pytest.raises(ValueError):
        WeightedHistoryRidge().fit(x, v, y, m & (torch.arange(4).reshape(1, 2, 2) != 0), w)


def test_constant_design_has_unpenalised_weighted_intercept():
    x = torch.ones(4, 2, 1, dtype=torch.float64)
    v = torch.ones_like(x, dtype=torch.bool)
    y = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
    m = torch.ones_like(y, dtype=torch.bool)
    w = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    model = WeightedHistoryRidge(alpha=100).fit(x, v, y, m, w)
    torch.testing.assert_close(model.predict(x, v), torch.full_like(y, 2.0))
    assert model.factorizations == 0


def test_neural_weighted_loss_preserves_float32_for_xpu_compatibility():
    p = torch.tensor([[[1.0]], [[3.0]]], requires_grad=True)
    w = torch.tensor([0.25, 0.75], dtype=torch.float64)
    loss = weighted_mse(p, torch.zeros_like(p), torch.ones_like(p, dtype=torch.bool), w)
    assert loss.dtype == torch.float32
    loss.backward()
    torch.testing.assert_close(p.grad, torch.tensor([[[0.5]], [[4.5]]]))
