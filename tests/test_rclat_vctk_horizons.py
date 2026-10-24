import numpy as np
import pytest
from bioacoustic_pipeline.diagnostics.rclat_vctk_horizons import eligible_origins, retained_losses


def test_complete_history_excludes_fir_startup_and_requires_all_retained_targets():
    origins = eligible_origins(48000)
    assert origins[0] == 32 and origins[-1] == 80
    assert eligible_origins(1000).size == 0


def test_disabled_40ms_values_cannot_change_retained_losses():
    p = np.ones((2, 3, 64))
    y = np.zeros_like(p)
    m = np.ones_like(p, dtype=bool)
    expected = retained_losses(p, y, m)
    p[:, 0] = np.nan
    y[:, 0] = np.inf
    m[:, 0] = False
    actual = retained_losses(p, y, m)
    assert set(actual) == {8, 16}
    for h in actual:
        np.testing.assert_array_equal(actual[h], expected[h])
    assert np.all(actual[8] == 1)


def test_observed_invalid_or_unsupported_retained_horizon_rejected():
    p = np.ones((2, 3, 64))
    y = np.zeros_like(p)
    m = np.ones_like(p, dtype=bool)
    y[0, 1, 0] = np.nan
    with pytest.raises(ValueError):
        retained_losses(p, y, m)
    m[0, 1, 0] = False
    assert np.isfinite(retained_losses(p, y, m)[8]).all()
    m[0, 2] = False
    with pytest.raises(ValueError):
        retained_losses(p, y, m)
