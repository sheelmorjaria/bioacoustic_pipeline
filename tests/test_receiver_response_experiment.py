"""Pre-implementation tests for grouped receiver-response comparisons."""
import numpy as np
import pytest

from bioacoustic_pipeline.contextual.receiver_response import evaluate_response_models


def examples():
    return [dict(event_id=f'{g}-{i}', group=str(g), outcome=float(g + i),
                 metadata=[float(i)], acoustics=[float(g-i)])
            for g in range(4) for i in range(3)]


def test_models_share_targets_and_never_fit_a_heldout_group():
    rows = examples()
    out = evaluate_response_models(rows)
    assert len(out['predictions']) == len(rows)
    assert {p['event_id'] for p in out['predictions']} == {r['event_id'] for r in rows}
    for fold in out['folds']:
        assert fold['heldout_group'] not in fold['fit_groups']
    for prediction in out['predictions']:
        assert set(prediction['predictions']) == {'metadata', 'metadata_acoustic'}
        assert all(np.isfinite(v) for v in prediction['predictions'].values())


def test_heldout_labels_cannot_change_their_own_predictions():
    rows = examples()
    first = evaluate_response_models(rows)
    changed = [dict(r, outcome=1e6) if r['group'] == '0' else r for r in rows]
    second = evaluate_response_models(changed)
    p = [r['predictions'] for r in first['predictions'] if r['group']=='0']
    q = [r['predictions'] for r in second['predictions'] if r['group']=='0']
    assert p == q


def test_missing_outcomes_and_insufficient_groups_fail_before_fitting():
    with pytest.raises(ValueError, match='groups'):
        evaluate_response_models([r for r in examples() if r['group'] in ['0','1']])
    bad = examples()
    bad[0]['outcome'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        evaluate_response_models(bad)
    bad = examples()
    bad[0]['event_id'] = bad[1]['event_id']
    with pytest.raises(ValueError, match='unique'):
        evaluate_response_models(bad)


def test_treatment_feature_is_not_reported_as_acoustics():
    out = evaluate_response_models(examples(), added_feature_kind='treatment')
    assert set(out['mse']) == {'metadata', 'metadata_treatment'}
    assert out['added_feature_kind'] == 'treatment'
