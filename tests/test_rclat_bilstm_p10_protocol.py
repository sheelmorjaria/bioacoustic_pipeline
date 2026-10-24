import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rclat_bilstm_p10_protocol import validate_contract  # noqa: E402

ROOT = Path("outputs/rclat_bilstm_p10_v1")


def test_frozen_protocol_population_budget_and_blocks():
    p = json.loads((ROOT / "preregistration.json").read_text())
    validate_contract(p)
    assert ((p["population"]["training_utterances"] + 3) // 4) * p["training"]["epochs"] == p[
        "training"
    ]["updates"]
    assert p["population"]["training_speakers"] == 251
    assert p["population"]["validation_speakers"] == 33
    assert p["training"]["internal_walltime_seconds_per_device"] == 43200
    assert p["secondary_grounding"]["selection_use"] == "prohibited"
    assert not p["runner_implemented"] and not p["scientific_training_started"]


@pytest.mark.parametrize("change", ["resume", "gate", "automatic_start"])
def test_reject_incompatible_scientific_contract(change):
    p = copy.deepcopy(json.loads((ROOT / "preregistration.json").read_text()))
    if change == "resume":
        p["training"]["resume"] = True
    elif change == "gate":
        p["learning_gate"]["speaker_macro_CER_strict_maximum"] = 0.8
    else:
        p["automatic_scientific_start"] = True
    with pytest.raises(AssertionError):
        validate_contract(p)
