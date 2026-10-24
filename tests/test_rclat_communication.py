import copy
import math

import pytest
import torch

from bioacoustic_pipeline.diagnostics.rclat_communication import (
    listener_targets,
    fit_script_baseline,
    speaker_macro_loss,
    crossed_interval,
    readiness,
    TemporalRequestModel,
)


def data():
    stimuli = [
        {"stimulus_id": "a", "source_filename": "a.wav", "speaker_id": "s1", "text_id": "t"},
        {"stimulus_id": "b", "source_filename": "b.wav", "speaker_id": "s2", "text_id": "t"},
    ]
    responses = [
        {"source_filename": "a.wav", "listener_id": "l1", "perceived_request": 1},
        {"source_filename": "a.wav", "listener_id": "l2", "perceived_request": 0},
        {"source_filename": "b.wav", "listener_id": "l1", "perceived_request": 1},
    ]
    return stimuli, responses


def test_targets_use_actual_listener_responses_not_production_intent():
    s, r = data()
    targets = listener_targets(s, r)
    assert targets["a"]["request_probability"] == 0.5 and targets["a"]["listeners"] == 2
    assert targets["b"]["request_probability"] == 1
    changed = copy.deepcopy(s)
    for row in changed:
        row["production_intent_private_not_prediction_target"] = "arbitrary"
    assert listener_targets(changed, r) == targets
    with pytest.raises(ValueError, match="duplicate"):
        listener_targets(s, r + [r[0]])
    with pytest.raises(ValueError, match="unknown"):
        listener_targets(s, r + [dict(r[0], source_filename="missing")])


def test_equal_stimulus_speaker_scoring_and_training_only_lexical_baseline():
    s, r = data()
    targets = listener_targets(s, r)
    baseline = fit_script_baseline(targets, ["a"], strength=1)
    assert baseline["scripts"]["t"] == 0.5
    assert speaker_macro_loss(s, r, {"a": 0.5, "b": 0.5}) == pytest.approx(math.log(2))
    with pytest.raises(ValueError):
        speaker_macro_loss(s, r, {"a": 2, "b": 0.5})


def test_crossed_interval_reproducible_and_paired():
    s, r = data()
    result = crossed_interval(
        s, r, {"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}, seed=7, replicates=100
    )
    assert result["difference"] == 0 and result["interval95"] == [0, 0]
    assert result == crossed_interval(
        s, r, {"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}, seed=7, replicates=100
    )


def test_capacity_matched_temporal_models_hide_fine_values_from_c1():
    torch.manual_seed(0)
    model = TemporalRequestModel(lexical_dim=3, acoustic_dim=2, steps=5, hidden=4)
    lexical = torch.randn(2, 3)
    summary = torch.randn(2, 2)
    trajectory = torch.randn(2, 5, 2)
    a = model(lexical, summary, trajectory, arm="C1")
    assert torch.equal(a, model(lexical, summary, trajectory + 10, arm="C1"))
    b = model(lexical, summary, trajectory, arm="C2")
    assert not torch.equal(b, model(lexical, summary, trajectory + 10, arm="C2"))
    b.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_readiness_does_not_implicitly_approve_missing_reviews_or_sources():
    report = readiness(wording={"ready": False}, source_use=False, methodology_frozen=False)
    assert not report["ready"] and len(report["blockers"]) == 3
    assert readiness(wording={"ready": True}, source_use=True, methodology_frozen=True)["ready"]


def test_responses_cannot_disagree_with_stimulus_speaker_metadata():
    s, r = data()
    with pytest.raises(ValueError, match="identity"):
        listener_targets(s, [dict(x, speaker_id="wrong") for x in r])


def test_stimulus_weights_do_not_follow_listener_counts():
    s = [
        {"stimulus_id": u, "source_filename": u, "speaker_id": speaker, "text_id": "t"}
        for u, speaker in [("a", "s1"), ("b", "s1"), ("c", "s2")]
    ]
    r = [{"source_filename": "a", "listener_id": str(i), "perceived_request": 1} for i in range(10)]
    r += [
        {"source_filename": "b", "listener_id": "0", "perceived_request": 0},
        {"source_filename": "c", "listener_id": "0", "perceived_request": 1},
    ]
    expected = ((-math.log(0.9) - math.log(0.1)) / 2 - math.log(0.5)) / 2
    assert speaker_macro_loss(s, r, {"a": 0.9, "b": 0.9, "c": 0.5}) == pytest.approx(expected)


def test_source_script_aliases_are_explicit_and_do_not_change_judgments():
    from bioacoustic_pipeline.diagnostics.rclat_communication import canonicalize_responses

    s, r = data()
    s[0]["text_id"] = "temperature"
    r[0]["text_id"] = "temp"
    r[1]["text_id"] = "temp"
    with pytest.raises(ValueError, match="identity"):
        listener_targets(s, r)
    fixed = canonicalize_responses(r, {"temp": "temperature"})
    assert r[0]["text_id"] == "temp" and fixed[0]["source_text_id"] == "temp"
    assert fixed[0]["perceived_request"] == r[0]["perceived_request"]
    assert listener_targets(s, fixed)["a"]["request_probability"] == 0.5
