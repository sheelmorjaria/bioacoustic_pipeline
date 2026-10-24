import pytest

from bioacoustic_pipeline.ingestion.request_protocol import (
    lexical_tokens,
    proposed_speaker_folds,
    wording_gate,
)


def meta(name):
    return {
        **{
            k: True
            for k in [
                "human_review",
                "independent_review",
                "source_scripts_hidden",
                "production_intentions_hidden",
                "listener_judgments_hidden",
                "model_transcripts_hidden",
            ]
        },
        "reviewer_id": name,
    }


def target():
    return [dict(review_id="r1", script_text="My car isn't starting.")]


def complete(text):
    return [dict(review_id="r1", status="completed", heard_text=text)]


def test_blank_reviews_never_certify_wording():
    result = wording_gate(target(), [], [], {}, {}, [])
    assert not result["ready"] and result["unresolved_stimuli"] == ["r1"]


def test_word_substitution_is_not_normalised_away():
    assert lexical_tokens("MY car isn’t starting!") == lexical_tokens("My car isn't starting.")
    result = wording_gate(
        target(),
        complete("My car is not starting"),
        complete("My car is not starting"),
        meta("a"),
        meta("b"),
        [],
    )
    assert not result["ready"] and result["verified_wording_mismatches"] == ["r1"]


def test_matching_transcripts_need_independent_humans():
    rows = complete("My car isn't starting")
    assert wording_gate(target(), rows, rows, meta("a"), meta("b"), [])["ready"]
    assert not wording_gate(target(), rows, rows, {}, {}, [])["ready"]
    with pytest.raises(ValueError, match="distinct"):
        wording_gate(target(), rows, rows, meta("a"), meta("a"), [])


def test_disagreement_requires_third_identified_human_and_evidence():
    a = complete("My car isn't starting")
    b = complete("My car isn't stopping")
    assert not wording_gate(target(), a, b, meta("a"), meta("b"), [])["ready"]
    decision = [
        dict(
            review_id="r1",
            status="resolved",
            adjudicator_id="c",
            heard_text="My car isn't starting",
            evidence="Independent listening confirmed starting",
            human_review=True,
            source_scripts_hidden=True,
            production_intentions_hidden=True,
        )
    ]
    assert wording_gate(target(), a, b, meta("a"), meta("b"), decision)["ready"]
    decision[0]["adjudicator_id"] = "a"
    assert not wording_gate(target(), a, b, meta("a"), meta("b"), decision)["ready"]


def test_folds_hold_entire_speakers_and_preserve_all_scripts():
    stimuli = [
        dict(stimulus_id=f"{s}_{t}_{c}", speaker_id=str(s), text_id=str(t))
        for s in range(18)
        for t in range(12)
        for c in range(2)
    ]
    folds = proposed_speaker_folds(stimuli)
    assert len(folds) == 18
    for fold in folds:
        train = set(fold["training_speakers"])
        val = set(fold["validation_speakers"])
        assert len(train) == 14 and len(val) == 3 and not train & val
        assert fold["held_out_speaker"] not in train | val
        assert (fold["training_stimuli"], fold["validation_stimuli"], fold["held_out_stimuli"]) == (
            336,
            72,
            24,
        )
    with pytest.raises(ValueError, match="paired"):
        proposed_speaker_folds(stimuli[:-1])
