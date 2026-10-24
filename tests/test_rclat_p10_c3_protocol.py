"""C3 scientific-policy preservation and fail-closed runtime prerequisites."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))
import rclat_p10_c3_train as current  # noqa: E402


def test_scientific_design_preserved_and_no_engineering_resume():
    prior = json.loads(Path("outputs/rclat_p10_corrected_v2/preregistration.json").read_text())
    new = json.loads((current.OUT / "preregistration.json").read_text())
    for key in [
        "selection",
        "frontends",
        "features",
        "architecture",
        "initialization",
        "training",
        "validation",
        "learning_gate",
        "source_population",
        "final_test",
    ]:
        assert new[key] == prior[key], key
    assert new["training"]["epochs"] == 30
    assert not new["automatic_scientific_start"]
    assert not new["predecessor"]["continuation_or_resume"]
    assert not new["stage2_ready"]


@pytest.mark.parametrize("capacity", [None, "8192"])
def test_wrong_capacity_is_rejected(monkeypatch, capacity):
    monkeypatch.delenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", raising=False)
    if capacity is not None:
        monkeypatch.setenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", capacity)
    with pytest.raises(ValueError, match="capacity16384"):
        current.check()


def test_verified_runtime_and_engineering_gate_without_training(monkeypatch):
    monkeypatch.setenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "16384")
    monkeypatch.setenv("ONEDNN_VERBOSE", "profile_create,filter=conv")
    monkeypatch.delenv("DNNL_PRIMITIVE_CACHE_CAPACITY", raising=False)
    spec, cohort = current.check()
    assert len(cohort["selected_training"]) == 2814
    assert len(cohort["selected_validation"]) == 2864
    assert spec["learning_gate"]["both_arms_required"]


def test_legacy_override_rejected(monkeypatch):
    monkeypatch.setenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "16384")
    monkeypatch.setenv("DNNL_PRIMITIVE_CACHE_CAPACITY", "8192")
    with pytest.raises(ValueError, match="Legacy"):
        current.check()
