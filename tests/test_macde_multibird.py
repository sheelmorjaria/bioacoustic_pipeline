import numpy as np
import pytest
from bioacoustic_pipeline.diagnostics.macde_multibird import bird_folds, fit_preprocessing, select_ridge


def test_outer_and_inner_birds_are_disjoint():
    birds = np.array(['a','a','b','c','d'])
    train, test, inner = bird_folds(birds,'d')
    assert set(birds[train]) == {'a','b','c'} and set(birds[test]) == {'d'}
    for fit, val in inner:
        assert not set(birds[fit]) & set(birds[val])
        assert 'd' not in set(birds[fit]) | set(birds[val])
    with pytest.raises(ValueError):
        bird_folds(np.array(['a','b','c']),'c')


def test_heldout_values_do_not_fit_scalers_or_reporting_scale():
    arrays = {'time_valid':np.ones((4,2),bool),'target':np.arange(8.).reshape(4,2)}
    for key in ['spectral','physical']:
        arrays[key] = np.arange(16.,dtype=float).reshape(4,2,2)
        arrays[key+'_valid'] = np.ones((4,2,2),bool)
    _,scales,target_sd = fit_preprocessing(arrays,np.array([0,1,2]))
    arrays['target'][3] = 1e12
    arrays['spectral'][3] = 1e12
    arrays['physical'][3] = 1e12
    _,other,other_sd = fit_preprocessing(arrays,np.array([0,1,2]))
    assert scales == other
    np.testing.assert_array_equal(target_sd,other_sd)


def test_baseline_selection_uses_inner_scores_and_fixed_ties():
    scores = {'R0':{1.:[2.,2.,2.],10.:[1.,1.,1.]},
              'R0_long':{1.:[1.,1.,1.],10.:[3.,3.,3.]}}
    result = select_ridge(scores)
    assert result['primary_R0_alpha'] == 10.
    assert result['selected_family'] == 'R0'
    assert result['alphas']['R0_long'] == 1.
