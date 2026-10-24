import pytest
import torch

from bioacoustic_pipeline.representation.rclat_phonology import PhonologicalCTC, joint_loss
from bioacoustic_pipeline.representation.rclat_simple_ctc import SimpleCTC


def provenance():
    return {
        "origin": "human_annotation",
        "independent_of_acoustic_model": True,
        "source_use_approved": True,
        "source_sha256": "a" * 64,
    }


def test_character_path_preserved_and_auxiliary_head_connected():
    torch.manual_seed(0)
    baseline = SimpleCTC()
    torch.manual_seed(0)
    model = PhonologicalCTC(["voiced", "sonorant"])
    x = torch.randn(2, 17, 64)
    m = torch.ones_like(x, dtype=torch.bool)
    n = torch.tensor([17, 13])
    c, f, lengths = model(x, m, n)
    assert torch.equal(c, baseline(x, m, n)[0])
    assert f.shape == (2, 5, 2) and lengths.tolist() == [5, 4]
    f.square().mean().backward()
    assert model.head.features.weight.grad is not None


def test_soft_reference_loss_masks_missing_values_and_preserves_baseline():
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    target = torch.tensor([[[1.0, 0.0], [0.5, float("nan")], [float("nan"), float("nan")]]])
    observed = torch.tensor([[[True, True], [True, False], [False, False]]])
    primary = torch.tensor(2.0, requires_grad=True)
    loss = joint_loss(primary, logits, target, observed, torch.tensor([2]), 0.2, provenance())
    assert float(loss.detach()) == pytest.approx(2 + 0.2 * torch.log(torch.tensor(2.0)).item())
    loss.backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad[0, 2].any()
    assert joint_loss(primary, None, None, None, None, 0, None) is primary


def test_auxiliary_supervision_cannot_be_model_defined_or_padded():
    x = torch.zeros(1, 2, 1, requires_grad=True)
    t = torch.zeros_like(x)
    mask = torch.ones_like(x, dtype=torch.bool)
    with pytest.raises(ValueError, match="independent"):
        joint_loss(
            x.sum(),
            x,
            t,
            mask,
            torch.tensor([2]),
            0.1,
            {**provenance(), "origin": "model_prediction"},
        )
    with pytest.raises(ValueError, match="padding"):
        joint_loss(x.sum(), x, t, mask, torch.tensor([1]), 0.1, provenance())
    with pytest.raises(ValueError, match="observed"):
        joint_loss(x.sum(), x, t, torch.zeros_like(mask), torch.tensor([2]), 0.1, provenance())


def test_inventory_is_not_a_bare_string_and_lengths_are_integral():
    with pytest.raises(ValueError, match="inventory"):
        PhonologicalCTC("voiced")
    x = torch.zeros(1, 2, 1)
    mask = torch.ones_like(x, dtype=torch.bool)
    with pytest.raises(ValueError, match="lengths"):
        joint_loss(x.sum(), x, x, mask, torch.tensor([1.5]), 0.1, provenance())
