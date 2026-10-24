import numpy as np
import pytest
from bioacoustic_pipeline.diagnostics.rclat_rtpe_inventory import (
    eligible_origins,
    select_origins,
    trailing_mel,
    recording_weights,
)


def test_origin_support_matches_rtpe_gap_contract():
    valid = np.ones((70, 3), bool)
    valid[35] = False
    origins = eligible_origins(valid, history=8, horizons=(4, 8, 16))
    expected = [t for t in range(7, 54) if not (t - 7 <= 35 <= t + 16)]
    assert origins.tolist() == expected
    valid[35, 0] = True
    assert len(eligible_origins(valid, history=8, horizons=(4, 8, 16))) == 47


def test_sampler_depends_only_on_identity_eligibility_and_frozen_salt():
    candidates = np.arange(31, 1000)
    a = select_origins("a", candidates, count=8, salt="fixed")
    assert len(a) == 8 and len(set(a)) == 8
    assert a == select_origins("a", candidates[::-1], count=8, salt="fixed")
    assert a != select_origins("b", candidates, count=8, salt="fixed")
    with pytest.raises(ValueError):
        select_origins("a", np.arange(3), count=8, salt="fixed")
    with pytest.raises(ValueError):
        select_origins("a", np.array([1, 1, 2]), count=2, salt="fixed")


def test_trailing_mel_matches_existing_frontend_on_synthetic_audio():
    from bioacoustic_pipeline.representation.rclat_ctc import acoustic_features

    x = np.random.default_rng(18).normal(0, 0.1, 2240)
    np.testing.assert_allclose(
        trailing_mel(x), acoustic_features(x, 16000)["mel"], rtol=1e-6, atol=1e-6
    )


def test_training_weights_equalise_speakers_and_recordings():
    rows = [dict(uid="a", speaker="1"), dict(uid="b", speaker="1"), dict(uid="c", speaker="2")]
    w = recording_weights(rows)
    assert w == dict(a=0.25, b=0.25, c=0.5)
    assert sum(w.values()) == 1
