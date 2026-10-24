import numpy as np
import pytest
from bioacoustic_pipeline.diagnostics.frozen_acoustic_transfer import frozen_scale, frozen_ridge


def test_frozen_scaler_does_not_adapt_to_target_distribution():
    state = {'mean':[2.,4.], 'scale':[2.,4.], 'support':[True,True]}
    x = np.array([[4.,8.],[100.,400.]])
    a, mask = frozen_scale(x,np.ones_like(x,bool),state)
    np.testing.assert_allclose(a,[[1.,1.],[49.,99.]])
    assert mask.all() and state['mean'] == [2.,4.]


def test_missing_inputs_remain_masked():
    state = {'mean':[0.,0.], 'scale':[1.,1.], 'support':[True,False]}
    x, mask = frozen_scale(np.array([[np.nan,7.]]),np.array([[False,True]]),state)
    assert not mask.any() and (x == 0).all()


def test_frozen_ridge_is_affine_and_rejects_unsupported_values():
    state = {'input_mean':np.array([2.]),'input_scale':np.array([2.]),
             'output_mean':np.array([10.]),'output_scale':np.array([3.]),
             'weight':np.array([[4.]]),'intercept':np.array([1.])}
    np.testing.assert_allclose(frozen_ridge(state,np.array([[4.]])),[[25.]])
    with pytest.raises(ValueError):
        frozen_ridge(state,np.array([[np.nan]]))
