import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from bioacoustic_pipeline.diagnostics.rclat_mask_operator import (
    shape_preserving_checks,
    reference_selection,
)


class NoIndex(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert str(func) != "aten.index.Tensor", "Candidate guard must not use boolean gathering"
        return func(*args, **(kwargs or {}))


def test_masked_nonfinite_ignored_but_observed_nonfinite_detected():
    x = torch.tensor([[1.0, float("nan"), float("inf")], [2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, False, False], [True, True, True]])
    with NoIndex():
        info = shape_preserving_checks(x, mask)
    assert info["and_valid"] and info["where_valid"]
    assert info["observed_counts"] == [1, 3] and info["observed_nonfinite_counts"] == [0, 0]
    x[1, 1] = float("nan")
    with NoIndex():
        info = shape_preserving_checks(x, mask)
    assert not info["and_valid"] and not info["where_valid"]
    assert info["observed_nonfinite_counts"] == [0, 1]


def test_nonempty_support_required_per_origin():
    x = torch.full((2, 3), float("nan"))
    mask = torch.zeros_like(x, dtype=torch.bool)
    result = shape_preserving_checks(x, mask)
    assert not result["and_valid"] and not result["where_valid"]
    x[1] = 1
    mask[1] = True
    result = shape_preserving_checks(x, mask)
    assert not result["and_valid"] and not result["where_valid"]


def test_reference_and_checks_agree_for_strided_and_offset_views():
    x = torch.arange(48.0).reshape(2, 3, 8)[..., ::2]
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, 0, 1] = False
    x[0, 0, 1] = float("nan")
    reference = reference_selection(x, mask)
    actual = shape_preserving_checks(x, mask)
    assert reference["expected_count"] == 23 and reference["valid"]
    assert actual["and_valid"] == reference["valid"] == actual["where_valid"]
    assert actual["observed_counts"] == reference["observed_counts"]
    with pytest.raises(ValueError):
        shape_preserving_checks(x, mask.float())


def test_layout_construction_preserves_values_and_requested_storage():
    from bioacoustic_pipeline.diagnostics.rclat_mask_operator import layout_copy

    x = torch.arange(24.0).reshape(2, 3, 4)
    m = torch.ones_like(x, dtype=torch.bool)
    for layout in ["contiguous", "strided", "offset7"]:
        a, b = layout_copy(x, m, layout, "cpu")
        torch.testing.assert_close(a, x, rtol=0, atol=0)
        assert torch.equal(b, m)
        if layout == "strided":
            assert a.stride()[-1] == 2 and b.stride()[-1] == 2
        if layout == "offset7":
            assert a.storage_offset() == 7 and b.storage_offset() == 7
