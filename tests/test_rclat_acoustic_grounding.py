import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rclat_acoustic_grounding import transform  # noqa: E402


def test_temporal_controls_preserve_length_masks_and_source():
    x = torch.arange(16.0).reshape(2, 4, 2)
    m = torch.ones_like(x, dtype=torch.bool)
    n = torch.tensor([3, 4])
    y, mask, lengths = transform(x, m, n, "reversed")
    assert torch.equal(y[0, :3], x[0, :3].flip(0))
    assert torch.equal(mask, m) and torch.equal(lengths, n)
    z, mask, lengths = transform(x, m, n, "zeroed")
    assert not z.any() and torch.equal(mask, m) and torch.equal(lengths, n)
    assert x.sum() > 0


def test_swapped_uses_only_donor_and_keeps_recipient_length():
    x = torch.zeros(1, 4, 2)
    m = torch.ones_like(x, dtype=torch.bool)
    n = torch.tensor([4])
    donor = torch.tensor([[[1., 2.], [3., 4.]]])
    y, _, lengths = transform(x, m, n, "swapped", (donor, torch.ones_like(donor,dtype=torch.bool), torch.tensor([2])))
    assert torch.equal(y, donor[:, [0, 0, 1, 1]])
    assert torch.equal(lengths, n) and not x.any()
