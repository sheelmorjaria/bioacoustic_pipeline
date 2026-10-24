import torch
from bioacoustic_pipeline.representation.rclat_rtpe import TemporalPredictionEngine
from bioacoustic_pipeline.representation.rclat_rtpe_pool import PoolTemporalPredictionEngine


def test_pool_means_preserve_forward_and_gradient_contract_with_missing_cells():
    torch.manual_seed(17)
    old = TemporalPredictionEngine(3, hidden=8, scales=(1, 4, 16), horizons=(4, 8, 16)).double()
    new = PoolTemporalPredictionEngine(3, hidden=8, scales=(1, 4, 16), horizons=(4, 8, 16)).double()
    new.load_state_dict(old.state_dict())
    x = torch.randn(2, 32, 3, dtype=torch.float64)
    m = torch.ones_like(x, dtype=torch.bool)
    m[:, 10, 2] = False
    x[~m] = float("nan")
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    pa = old(a, m)["forecast"]
    pb = new(b, m)["forecast"]
    torch.testing.assert_close(pa, pb, rtol=1e-12, atol=1e-12)
    pa.square().mean().backward()
    pb.square().mean().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(b.grad).all() and not b.grad[~m].any()


def test_pool_model_is_exactly_prefix_causal():
    torch.manual_seed(21)
    model = PoolTemporalPredictionEngine(3, hidden=8, scales=(1, 4, 16), horizons=(4, 8, 16))
    x = torch.randn(2, 32, 3)
    m = torch.ones_like(x, dtype=torch.bool)
    a = model(x, m)["forecast"]
    x[:, 20:] = 1000
    b = model(x, m)["forecast"]
    torch.testing.assert_close(a[:, :20], b[:, :20], rtol=0, atol=0)
