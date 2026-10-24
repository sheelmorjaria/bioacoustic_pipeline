import numpy as np

from bioacoustic_pipeline.diagnostics.macde_controls import history_controls


def test_controls_preserve_frames_recent_states_and_summary():
    history = np.arange(52 * 12).reshape(52, 12).astype(float)
    summary, ordered, shuffled = history_controls(history, 17)
    shuffled = shuffled.reshape(52, 12)
    np.testing.assert_array_equal(shuffled[-2:], history[-2:])
    np.testing.assert_array_equal(np.sort(shuffled[:, 0]), history[:, 0])
    np.testing.assert_allclose(summary, np.r_[history.mean(0), history.std(0)])
    np.testing.assert_array_equal(ordered, history.ravel())
    assert not np.array_equal(shuffled[:-2], history[:-2])
    np.testing.assert_array_equal(history_controls(history, 17)[2], shuffled.ravel())
    assert not np.array_equal(history_controls(history, 18)[2], shuffled.ravel())


def test_controls_reject_incomplete_or_invalid_history():
    import pytest
    with pytest.raises(ValueError):
        history_controls(np.ones((51, 12)), 0)
    with pytest.raises(ValueError):
        history_controls(np.full((52, 12), np.nan), 0)
