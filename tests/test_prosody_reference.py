import pytest

from bioacoustic_pipeline.ingestion.prosody_reference import (
    extract_request_judgments,
    request_distributions,
)


def row(response="Non-Request", condition="request", subject="listener1"):
    return dict(
        speaker="7",
        condition=condition,
        topic="office",
        subject=subject,
        response=response,
        button_pressed={"Request": "0", "Non-Request": "1", "": ""}[response],
    )


def test_listener_perception_overrides_no_production_intent():
    judgments, counts = extract_request_judgments([row()])
    assert judgments[0]["perceived_request"] == 0
    assert counts["actual_critical_responses"] == 1


def test_no_response_never_filled_using_intent_or_accuracy_label():
    missing = row("")
    missing["answer"] = "Request"
    missing["correct"] = "1"
    judgments, counts = extract_request_judgments([missing])
    assert not judgments and counts["critical_rows_without_response"] == 1


def test_disagreement_and_unique_listeners_preserved():
    judgments, _ = extract_request_judgments([row(), row("Request", subject="listener2")])
    distribution = request_distributions(judgments)["7_request_office.wav"]
    assert distribution["perceived_distribution"] == {"request": 0.5, "nonrequest": 0.5}
    assert distribution["independent_listeners"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        extract_request_judgments([row(), row()])


def test_mismatched_button_and_practice_not_valid_targets():
    wrong = row()
    wrong["button_pressed"] = "0"
    with pytest.raises(ValueError, match="mapping"):
        extract_request_judgments([wrong])
    practice = row()
    practice["speaker"] = ""
    judgments, counts = extract_request_judgments([practice])
    assert not judgments and counts["noncritical_or_practice_rows"] == 1
