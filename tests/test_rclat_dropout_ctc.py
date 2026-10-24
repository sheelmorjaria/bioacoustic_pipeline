import torch

from bioacoustic_pipeline.representation.rclat_dropout_ctc import DropoutCTC


def models():
    torch.manual_seed(0)
    a = DropoutCTC(0.0)
    torch.manual_seed(0)
    b = DropoutCTC(0.1)
    return a, b


def test_identical_initial_parameters_and_evaluation():
    a, b = models()
    assert a.state_dict().keys() == b.state_dict().keys()
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert sum(p.numel() for p in a.parameters()) == 781725
    x = torch.randn(2, 33, 64)
    m = torch.ones_like(x, dtype=torch.bool)
    n = torch.tensor([17, 33])
    a.eval()
    b.eval()
    assert torch.equal(a(x, m, n)[0], b(x, m, n)[0])


def test_dropout_active_only_during_training_with_connected_gradients():
    _, model = models()
    model.train()
    x = torch.randn(2, 33, 64)
    m = torch.ones_like(x, dtype=torch.bool)
    n = torch.tensor([17, 33])
    a = model(x, m, n)[0]
    b = model(x, m, n)[0]
    assert not torch.equal(a, b)
    a.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
