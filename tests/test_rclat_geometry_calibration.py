import pytest
from bioacoustic_pipeline.diagnostics.rclat_geometry_calibration import (
    extract_geometry,
    simulate_geometry,
)


def fixture():
    stimuli = [
        dict(stimulus_id=str(i), source_filename=str(i), speaker_id="s", text_id=str(i))
        for i in range(4)
    ]
    responses = [
        dict(source_filename=str(i), listener_id="l", perceived_request=object()) for i in range(4)
    ]
    return stimuli, responses


def test_whitelist_and_exact_edges():
    s, r = fixture()
    g = extract_geometry(s, r)
    assert all(set(e) == {"source_filename", "listener_id"} for e in g["edges"])
    d = simulate_geometry(g, dict(speaker_sd=0.8, listener_sd=0.8, alternative=False), seed=3)
    assert extract_geometry(d["stimuli"], d["responses"]) == g
    assert d == simulate_geometry(
        g, dict(speaker_sd=0.8, listener_sd=0.8, alternative=False), seed=3
    )


def test_reject_duplicate_unknown_and_unbalanced():
    s, r = fixture()
    with pytest.raises(ValueError):
        extract_geometry(s, r + r[:1])
    with pytest.raises(ValueError):
        extract_geometry(s, r + [dict(source_filename="bad", listener_id="l")])
    with pytest.raises(ValueError):
        extract_geometry(s[:3], r[:3])
