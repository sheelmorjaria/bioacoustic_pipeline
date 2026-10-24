import torch

from bioacoustic_pipeline.representation.rclat_simple_ctc import SimpleCTC


def test_lengths_padding_and_connected_gradients():
    torch.manual_seed(0)
    model = SimpleCTC()
    x = torch.randn(2, 33, 64)
    mask = torch.ones_like(x, dtype=torch.bool)
    lengths = torch.tensor([17, 33])
    y, n = model(x, mask, lengths)
    assert n.tolist() == [5, 9]
    changed = x.clone()
    changed[0, 17:] = 1e6
    z, _ = model(changed, mask, lengths)
    assert torch.equal(y, z)
    alone, _ = model(x[:1, :17], mask[:1, :17], lengths[:1])
    assert torch.allclose(alone[0], y[0, :5], atol=1e-6)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
