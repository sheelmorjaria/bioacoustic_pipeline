"""Fixed, grouped continuous-outcome comparison for receiver response data.

No hyperparameter search. All preprocessing and regression are fitted inside
leave-one-independent-group-out folds. Callers must freeze outcome provenance,
cohort and features before using this evaluator on scientific material.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_response_models(examples, *, bootstrap_draws=2000, seed=0,
                             added_feature_kind='acoustic'):
    if added_feature_kind not in ('acoustic', 'treatment'):
        raise ValueError('Unknown added feature kind')
    added_name = 'metadata_' + added_feature_kind
    groups = sorted({r['group'] for r in examples})
    if len(groups) < 3:
        raise ValueError('At least three independent groups required')
    if len({r['event_id'] for r in examples}) != len(examples):
        raise ValueError('Event identifiers must be unique')
    y = np.array([r['outcome'] for r in examples], dtype=float)
    metadata = np.asarray([r['metadata'] for r in examples], dtype=float)
    acoustic = np.asarray([r['acoustics'] for r in examples], dtype=float)
    if metadata.ndim != 2 or acoustic.ndim != 2 or not acoustic.shape[1]:
        raise ValueError('Rectangular nonempty acoustic features required')
    if not all(np.isfinite(x).all() for x in (y, metadata, acoustic)):
        raise ValueError('Outcomes and predictors must be finite; freeze exclusions first')
    matrices = {'metadata': metadata, added_name: np.column_stack([metadata, acoustic])}
    group_array = np.array([r['group'] for r in examples])
    predictions = {k: np.zeros(len(y)) for k in matrices}
    folds = []
    for group in groups:
        test = group_array == group
        train = ~test
        folds.append({'heldout_group': group, 'fit_groups': sorted(set(group_array[train])),
                      'fit_events': int(train.sum()), 'heldout_events': int(test.sum())})
        for key, x in matrices.items():
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(x[train], y[train])
            predictions[key][test] = model.predict(x[test])
    loss = {k: (p-y)**2 for k, p in predictions.items()}
    improvement = loss['metadata'] - loss[added_name]
    sums = np.array([improvement[group_array == g].sum() for g in groups])
    sizes = np.array([(group_array == g).sum() for g in groups])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), (bootstrap_draws, len(groups)))
    draws = sums[sampled].sum(1) / sizes[sampled].sum(1)
    return {'endpoint': 'supplied independently validated continuous receiver response',
            'metric': 'event-weighted squared error; native outcome units squared',
            'mse': {k: float(v.mean()) for k, v in loss.items()},
            'added_feature_kind': added_feature_kind,
            'delta_metadata_minus_added': {'point': float(improvement.mean()),
                                              'ci95': np.quantile(draws, [.025,.975]).tolist(),
                                              'clusters': len(groups), 'draws': bootstrap_draws},
            'folds': folds,
            'predictions': [{'event_id': r['event_id'], 'group': r['group'], 'observed': float(y[i]),
                             'predictions': {k: float(p[i]) for k, p in predictions.items()}}
                            for i,r in enumerate(examples)],
            'limits': ['OOF models share training material; cluster bootstrap interval is descriptive',
                       'no causal or semantic interpretation from prediction alone']}
