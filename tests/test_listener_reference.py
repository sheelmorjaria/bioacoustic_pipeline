import pytest

from bioacoustic_pipeline.ingestion.listener_reference import listener_reference_gate


def stimulus(uid="one", split="train", speaker="s", text="t"):
    return dict(
        stimulus_id=uid,
        split=split,
        speaker_id=speaker,
        text_id=text,
        intended_condition="question",
    )


def response(listener, uid="one", interpretation="statement"):
    return dict(
        stimulus_id=uid,
        listener_id=listener,
        status="completed",
        interpretation=interpretation,
        heard_text="You are coming",
    )


def metadata():
    return dict(
        reference_custodian_id="human_custodian",
        human_judgments=True,
        independent_of_models=True,
        production_intentions_hidden=True,
        reference_transcripts_hidden=True,
        production_participants_excluded=True,
        collection_authorization_recorded=True,
    )


def test_intentions_never_supply_missing_human_reference():
    result = listener_reference_gate([stimulus()], [], metadata(), 2)
    assert not result["ready"]
    assert result["references"]["one"]["distribution"] is None
    assert result["completed_judgments"] == 0


def test_listener_disagreement_preserved_not_replaced_by_intention():
    result = listener_reference_gate(
        [stimulus()], [response("a"), response("b", interpretation="uncertain")], metadata(), 2
    )
    assert result["ready"]
    assert result["references"]["one"]["distribution"] == dict(
        statement=0.5, question=0.0, uncertain=0.5
    )
    assert not result["majority_agreement_filter_applied"]


def test_pending_evidence_and_duplicate_responses_rejected():
    with pytest.raises(ValueError, match="Pending"):
        listener_reference_gate(
            [stimulus()],
            [dict(stimulus_id="one", status="pending", interpretation="question")],
            metadata(),
            2,
        )
    with pytest.raises(ValueError, match="Duplicate listener"):
        listener_reference_gate([stimulus()], [response("a"), response("a")], metadata(), 2)


def test_cross_partition_identity_leakage_rejected():
    with pytest.raises(ValueError, match="leakage"):
        listener_reference_gate([stimulus(), stimulus("two", "evaluation")], [], metadata(), 2)
    items = [stimulus(), stimulus("two", "evaluation", "other", "different")]
    with pytest.raises(ValueError, match="panels"):
        listener_reference_gate(items, [response("a"), response("a", "two")], metadata(), 2)


def test_listener_cannot_compare_two_versions_of_same_pair():
    with pytest.raises(ValueError, match="multiple versions"):
        listener_reference_gate(
            [stimulus(), stimulus("two")], [response("a"), response("a", "two")], metadata(), 2
        )


def test_human_provenance_is_mandatory_even_with_complete_choices():
    result = listener_reference_gate([stimulus()], [response("a"), response("b")], {}, 2)
    assert not result["ready"] and not result["independence_attested"]
