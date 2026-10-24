import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rclat_bilstm_p10_train import select_checkpoint, learning_gate, score  # noqa: E402


def test_selection_requires_all_epochs_and_earliest_tie():
    history = [
        {"epoch": e, "speaker_macro_cer": 0.7, "nonempty_fraction": 1.0} for e in range(1, 31)
    ]
    history[6]["speaker_macro_cer"] = 0.4
    history[18]["speaker_macro_cer"] = 0.4
    assert select_checkpoint(list(reversed(history)))["epoch"] == 7
    with pytest.raises(ValueError):
        select_checkpoint(history[:29])


def test_gate_strict_cer_inclusive_nonempty():
    assert learning_gate({"speaker_macro_cer": 0.499, "nonempty_fraction": 0.95})
    assert not learning_gate({"speaker_macro_cer": 0.5, "nonempty_fraction": 1})
    assert not learning_gate({"speaker_macro_cer": 0.1, "nonempty_fraction": 0.949})


def test_prediction_error_decomposition_keeps_reference_and_native_length():
    p = score({"uid": "u", "speaker": "s", "transcript": "ABC DEF"}, "ABC X", 99)
    assert p["reference_characters"] == 6 and p["native_length"] == 99
    assert p["character_errors"] == 3
    assert (
        sum(p["character_decomposition"][c] for c in ["substitutions", "deletions", "insertions"])
        == p["character_errors"]
    )
