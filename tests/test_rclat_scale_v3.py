import torch
import pytest
from bioacoustic_pipeline.representation.rclat_scale_v3 import ScaleCTC


def test_subsampling_lengths_and_padding_do_not_leak():
    torch.set_num_threads(2)
    model = ScaleCTC(4, width=16, layers=2, heads=2, expansion=32, kernel=3).eval()
    x = torch.randn(2, 37, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        logits, lengths = model(x, mask, torch.tensor([17, 37]))
        one, length = model(x[:1, :17], mask[:1, :17], torch.tensor([17]))
    assert lengths.tolist() == [5, 10] and length.tolist() == [5]
    torch.testing.assert_close(logits[:1, :5], one, atol=2e-6, rtol=2e-6)


def test_common_encoder_capacity_and_initialization():
    a = ScaleCTC(64, width=16, layers=1, heads=2, expansion=32, kernel=3)
    b = ScaleCTC(76, width=16, layers=1, heads=2, expansion=32, kernel=3)
    for module in ["subsample", "blocks", "head"]:
        left = getattr(a, module).state_dict()
        right = getattr(b, module).state_dict()
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)


def test_invalid_lengths_and_masked_nonfinite_features():
    model = ScaleCTC(4, width=16, layers=1, heads=2, expansion=32, kernel=3)
    x = torch.randn(1, 20, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    with pytest.raises(ValueError, match="length"):
        model(x, mask, torch.tensor([21]))
    x[0, 0, 0] = float("nan")
    mask[0, 0, 0] = False
    logits, lengths = model(x, mask, torch.tensor([20]))
    assert torch.isfinite(logits).all() and lengths.tolist() == [5]


def test_explicit_attention_matches_reference_cpu_forward():
    from bioacoustic_pipeline.representation.rclat_scale import ScaleCTC as Reference

    old = Reference(4, width=16, layers=1, heads=2, expansion=32, kernel=3).eval()
    new = ScaleCTC(4, width=16, layers=1, heads=2, expansion=32, kernel=3).eval()
    new.load_state_dict(old.state_dict())
    x = torch.randn(2, 37, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    lengths = torch.tensor([17, 37])
    with torch.no_grad():
        a, al = old(x, mask, lengths)
        b, bl = new(x, mask, lengths)
    assert al.tolist() == bl.tolist()
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
