import numpy as np
import pytest
from bioacoustic_pipeline.diagnostics.rclat_rtpe_protocol import (
    paired_speaker_summary,
    select_minimum,
)


def test_paired_uncertainty_resamples_whole_speakers_and_is_reproducible():
    a = {str(i): float(i + 2) for i in range(8)}
    b = {str(i): float(i + 1) for i in range(8)}
    result = paired_speaker_summary(a, b, seed=9, replicates=200)
    assert result["difference"] == 1 and result["interval95"] == [1, 1]
    assert result["development_utility_passed"]
    assert result == paired_speaker_summary(a, b, seed=9, replicates=200)
    equal = paired_speaker_summary(a, a, seed=9, replicates=200)
    assert not equal["development_utility_passed"]
    assert not paired_speaker_summary(b, a, seed=9, replicates=200)["development_utility_passed"]


def test_uncertainty_rejects_nonpaired_nonfinite_and_insufficient_groups():
    with pytest.raises(ValueError):
        paired_speaker_summary({"a": 1}, {"b": 2}, seed=1, replicates=20)
    with pytest.raises(ValueError):
        paired_speaker_summary({"a": np.nan, "b": 1}, {"a": 1, "b": 2}, seed=1, replicates=20)
    with pytest.raises(ValueError):
        paired_speaker_summary({"a": 1}, {"a": 2}, seed=1, replicates=20)


def test_selection_preserves_declared_order_for_exact_ties():
    assert select_minimum([(0.1, 0.5), (1.0, 0.5), (10.0, 0.6)]) == 0.1
    assert select_minimum([(250, 0.7), (500, 0.5), (750, 0.5)]) == 500
    with pytest.raises(ValueError):
        select_minimum([(1, float("nan"))])
