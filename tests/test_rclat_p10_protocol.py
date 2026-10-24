"""Meaningful cohort/CTC constraints for the new, untrained P10 protocol."""

import importlib.util
from pathlib import Path
import pytest


spec = importlib.util.spec_from_file_location("p10", Path("scripts/rclat_p10_protocol.py"))
p10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p10)


def test_subset_deterministic_whole_utterances_and_budget():
    rows = [{"uid": str(i), "samples": 17 + i} for i in range(30)]
    selected = p10.select_ten_hours(rows, 100)
    assert selected == p10.select_ten_hours(list(reversed(rows)), 100)
    assert sum(r["samples"] for r in selected) <= 100
    assert all(r in rows for r in selected)
    leftover = 100 - sum(r["samples"] for r in selected)
    assert all(r["samples"] > leftover for r in rows if r not in selected)


def test_ctc_repeat_budget_after_actual_subsampling():
    # AA needs three CTC positions, unlike AB which needs two.
    assert p10.ctc_support(1760, "AB")[0]
    assert not p10.ctc_support(1760, "AA")[0]
    assert p10.ctc_support(1920, "AA")[0]
    with pytest.raises(ValueError, match="unsupported"):
        p10.ctc_support(640, "")
