import pytest
import torch

from bioacoustic_pipeline.diagnostics.rclat_guard_matrix import guard_fixture, schedule
from bioacoustic_pipeline.diagnostics.rclat_mask_operator import reference_selection
from bioacoustic_pipeline.diagnostics.rclat_mask_stages import worker_case


def test_schedule_is_complete_and_has_no_native_index_workers():
    rows = schedule()
    assert len(rows) == 132
    assert len({r["name"] for r in rows}) == 132
    assert {r["operation"] for r in rows} == {"and", "where"}
    assert {r["shape"] for r in rows} == {(4, 3, 64), (128, 3, 64)}
    assert {r["layout"] for r in rows} == {"contiguous", "strided", "offset7"}


@pytest.mark.parametrize(
    "pattern,valid",
    [
        ("finite", True),
        ("masked_nan", True),
        ("masked_posinf", True),
        ("masked_neginf", True),
        ("observed_nan", False),
        ("observed_posinf", False),
        ("observed_neginf", False),
        ("empty_first_origin", False),
        ("empty_all", False),
        ("sparse_one_per_origin", True),
        ("mixed_observed_nan_masked_inf", False),
    ],
)
def test_independent_reference_encodes_expected_semantics(pattern, valid):
    x, m = guard_fixture((4, 3, 64), pattern)
    ref = reference_selection(x, m)
    assert ref["valid"] == valid
    if pattern == "sparse_one_per_origin":
        assert ref["observed_counts"] == [1] * 4
    if pattern == "empty_first_origin":
        assert ref["observed_counts"][0] == 0 and ref["expected_count"] > 0
    if pattern.startswith("observed_"):
        assert ref["observed_nonfinite_counts"][0] == 1


def test_staged_predicates_reject_observed_nan_on_verified_strided_cpu_inputs(tmp_path):
    x, m = guard_fixture((4, 3, 64), "observed_nan")
    for op in ["and", "where"]:
        result = worker_case(x, m, "strided", op, tmp_path / op, device="cpu")
        assert result["status"] == "PASS" and result["input_roundtrip_verified"]
        assert result["verdict"] is False and result["reference_verdict"] is False
    assert torch.isnan(x.flatten()[0])


def test_staged_candidate_uses_no_torch_boolean_index(tmp_path):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoIndex(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert str(func) != "aten.index.Tensor"
            return func(*args, **(kwargs or {}))

    x, m = guard_fixture((4, 3, 64), "mixed_observed_nan_masked_inf")
    with NoIndex():
        result = worker_case(x, m, "offset7", "and", tmp_path / "worker", device="cpu")
    assert result["status"] == "PASS" and result["verdict"] is False
