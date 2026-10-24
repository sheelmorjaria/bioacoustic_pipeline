"""Check the prospectively fixed single-policy intervention."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rclat_ctc_lr_diagnostic import ORIGINAL_RATE, learning_rate  # noqa: E402


def test_common_prefix_and_original_policy():
    for step in range(1, 2001):
        assert learning_rate(step, "original") == ORIGINAL_RATE(step, 21120)
        if step <= 250:
            assert learning_rate(step, "capped250") == ORIGINAL_RATE(step, 21120)


def test_cap_and_invalid_policy():
    cap = ORIGINAL_RATE(250, 21120)
    assert cap == pytest.approx(0.0005005)
    for step in range(251, 2001):
        assert learning_rate(step, "capped250") == cap
    with pytest.raises(ValueError):
        learning_rate(1, "unknown")
