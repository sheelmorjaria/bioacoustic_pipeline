import torch

from bioacoustic_pipeline.representation.rclat_ctc import ctc_objective, encode_text
from bioacoustic_pipeline.representation.rclat_scale_v3 import ScaleCTC


def test_explicit_attention_ctc_backward_is_finite_with_padding():
    torch.set_num_threads(2)
    model = ScaleCTC(4, width=16, layers=2, heads=2, expansion=32, kernel=3)
    features = torch.randn(2, 37, 4)
    mask = torch.ones_like(features, dtype=torch.bool)
    features[0, 17:] = float("nan")
    logits, lengths = model(features, mask, torch.tensor([17, 37]))
    loss = ctc_objective(logits, lengths, [encode_text("A B"), encode_text("THE")])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.square().sum()) for p in model.parameters()) > 0
