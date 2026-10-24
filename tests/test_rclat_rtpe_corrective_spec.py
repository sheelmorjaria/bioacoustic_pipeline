import copy
import pytest
from bioacoustic_pipeline.diagnostics.rclat_rtpe_corrective_spec import corrective_protocol


def test_corrective_spec_preserves_every_original_scientific_field():
    original = {
        "seed_pairs": [{"initialization": 101, "sampler": 2026102311}],
        "model_budget_per_seed": 5000,
        "replication_criterion": "frozen gate",
        "status": "old",
        "frozen_utc": "old",
        "study": "old",
    }
    before = copy.deepcopy(original)
    new = corrective_protocol(original, "now", "abc")
    assert original == before
    for key in original.keys() - {"study", "status", "frozen_utc"}:
        assert new[key] == original[key]
    assert new["correction"]["scientific_execution_authorized"] is False
    assert new["correction"]["production_guard"] == "WHERE"
    assert new["correction"]["predecessor_protocol_sha256"] == "abc"
    assert new["correction"]["runner_status"] == "PENDING_SEPARATE_BINDING_AND_VERIFICATION"


def test_incomplete_predecessor_not_accepted():
    with pytest.raises(ValueError):
        corrective_protocol({"model_budget_per_seed": 5000}, "now", "abc")
