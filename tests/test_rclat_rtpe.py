"""RTPE contracts: causality, support, matched history and independent targets."""

import pytest
import torch

from bioacoustic_pipeline.representation.rclat_rtpe import (
    TemporalPredictionEngine,
    HistoryRidge,
    forecast_windows,
    masked_mse,
    permute_history,
    order_loss,
    validate_group_split,
)


def batch():
    torch.manual_seed(3)
    x = torch.randn(2, 12, 3)
    return x, torch.ones_like(x, dtype=torch.bool)


def test_prefix_causality_and_missing_values():
    x, m = batch()
    model = TemporalPredictionEngine(3, hidden=8, scales=(1, 3), horizons=(1, 2), symbols=5).eval()
    a = model(x, m)
    changed = x.clone()
    changed[:, 7:] = 1000
    b = model(changed, m)
    torch.testing.assert_close(a["forecast"][:, :7], b["forecast"][:, :7], rtol=0, atol=0)
    torch.testing.assert_close(a["symbols"][:, :7], b["symbols"][:, :7], rtol=0, atol=0)
    assert a["forecast"].shape == (2, 12, 2, 3)
    m[:, 2, 1] = False
    x[:, 2, 1] = float("nan")
    a = model(x, m)["forecast"]
    x[:, 2, 1] = 999
    torch.testing.assert_close(a, model(x, m)["forecast"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="finite"):
        model(x * float("nan"), m)


def test_windows_never_cross_gaps_or_recording_boundaries():
    x = torch.arange(36.0).reshape(12, 3)
    m = torch.ones_like(x, dtype=torch.bool)
    m[5] = False
    w = forecast_windows(x, m, history=3, horizons=(1, 2))
    assert w["origins"].tolist() == [2, 8, 9]
    for i, t in enumerate(w["origins"].tolist()):
        assert torch.equal(w["inputs"][i], x[t - 2 : t + 1])
        assert torch.equal(w["targets"][i], x[torch.tensor([t + 1, t + 2])])
    m[:] = False
    with pytest.raises(ValueError, match="eligible"):
        forecast_windows(x, m, history=3, horizons=(1,))


def test_loss_ignores_missing_targets_without_nan_gradients():
    p = torch.zeros(1, 2, requires_grad=True)
    y = torch.tensor([[2.0, float("nan")]])
    m = torch.tensor([[True, False]])
    loss = masked_mse(p, y, m)
    assert loss.item() == 4
    loss.backward()
    assert p.grad.tolist() == [[-4.0, 0.0]]
    with pytest.raises(ValueError):
        masked_mse(p, y, ~m)


def test_ridge_has_same_ordered_history_and_target_support():
    torch.manual_seed(9)
    x = torch.randn(40, 4, 2)
    m = torch.ones_like(x, dtype=torch.bool)
    y = (2 * x[:, 0, 0] - x[:, -1, 1]).reshape(-1, 1, 1)
    ridge = HistoryRidge(alpha=1e-6).fit(
        x[:30], m[:30], y[:30], torch.ones_like(y[:30], dtype=torch.bool)
    )
    pred = ridge.predict(x[30:], m[30:])
    assert masked_mse(pred, y[30:], torch.ones_like(pred, dtype=torch.bool)).item() < 1e-8
    with pytest.raises(ValueError):
        ridge.predict(x[:, :3], m[:, :3])


def test_order_controls_move_masks_and_do_not_self_certify_artifacts():
    x, m = batch()
    m[:, 1, 0] = False
    p = torch.arange(11, -1, -1)
    a, b = permute_history(x, m, p)
    assert torch.equal(a, x[:, p]) and torch.equal(b, m[:, p])
    with pytest.raises(ValueError):
        permute_history(x, m, torch.zeros(12, dtype=torch.long))
    with pytest.raises(ValueError, match="audit"):
        order_loss(torch.zeros(2), torch.ones(2), control_audit=None)


def test_group_split_blocks_recording_and_individual_leakage():
    good = [
        dict(recording="r1", individual="i1", split="train"),
        dict(recording="r2", individual="i2", split="test"),
    ]
    validate_group_split(good)
    with pytest.raises(ValueError):
        validate_group_split(good + [dict(recording="r3", individual="i1", split="test")])
    with pytest.raises(ValueError):
        validate_group_split(good + [dict(recording="r1", individual="i3", split="validation")])


def test_scaler_uses_training_observations_only():
    from bioacoustic_pipeline.representation.rclat_rtpe import TrainingScaler

    x = torch.tensor([[1.0, float("nan")], [3.0, 10.0], [5.0, 14.0]])
    m = torch.isfinite(x)
    scaler = TrainingScaler().fit(x, m)
    assert scaler.mean.tolist() == [3.0, 12.0]
    mean = scaler.mean.clone()
    z = scaler.transform(x, m)
    assert torch.isfinite(z).all() and z[0, 1] == 0
    scaler.transform(torch.full_like(x, 1000), torch.ones_like(m))
    assert torch.equal(mean, scaler.mean)


def test_behavior_is_blocked_and_order_auxiliary_gradients_are_valid():
    from bioacoustic_pipeline.representation.rclat_rtpe import BehavioralReadout

    with pytest.raises(ValueError, match="Independent"):
        BehavioralReadout(8, 2, reference={})
    logits = torch.tensor([1.0, -1.0], requires_grad=True)
    loss = order_loss(
        logits,
        torch.tensor([1.0, 0.0]),
        control_audit={"artifact_matching_approved": True, "audit_sha256": "a" * 64},
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()
