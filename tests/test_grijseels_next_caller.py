"""Mechanics checks; no scientific outcome thresholds or model selection."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    'grijseels_next_caller', Path(__file__).resolve().parents[1] / 'scripts/grijseels_next_caller.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def samples(events):
    rows, _ = study.event_windows(events)
    for row in rows:
        row.update(candidates=(0, 1), configuration=0)
    return rows


def test_ties_reset_context_and_cannot_be_targets():
    rows, dropped = study.event_windows([(0, 2, 0), (1, 2, 0), (1, 3, 1),
                                         (4, 6, 1), (5, 7, 0), (9, 10, 1)])
    assert [r['current_onset'] for r in rows] == [4, 5]
    assert rows[0]['history'] == [(4, 6, 1)]
    assert rows[0]['overlap']
    assert dropped['singleton_before_tie'] == 1
    assert dropped['events_in_tied_groups'] == 2


def test_prediction_inputs_are_causal_and_exclude_offsets():
    first = [(0, 8, 0), (3, 5, 1), (10, 11, 0), (20, 21, 1)]
    changed = [(0, 2, 0), (3, 4, 1), (100, 101, 1), (200, 201, 0)]
    a = study.input_arrays(samples(first), 3)
    b = study.input_arrays(samples(changed), 3)
    for key in ('tokens', 'lengths', 'summaries', 'current', 'configuration', 'mask'):
        np.testing.assert_array_equal(a[key][:2], b[key][:2])
    assert a['target'][1] != b['target'][1]


def test_unavailable_callers_have_exactly_zero_probability():
    rows = samples([(0, 1, 0), (3, 4, 1), (10, 11, 0)])
    arrays = study.input_arrays(rows, 3)
    data = {k: torch.from_numpy(v) for k, v in arrays.items()}
    for kind in ('M1', 'M2', 'M3'):
        model = study.build_model(kind, 3, 1, arrays['summaries'].shape[1])
        p = study.probabilities(model, data, np.arange(len(rows)))
        np.testing.assert_array_equal(p[:, 2], 0)
        np.testing.assert_allclose(p.sum(1), 1)


def test_standardization_does_not_use_heldout_rows():
    arrays = study.input_arrays(samples([(0, 1, 0), (3, 4, 1), (10, 11, 0), (20, 21, 1)]), 3)
    other = {k: v.copy() for k, v in arrays.items()}
    other['summaries'][-1] = 100000
    other['tokens'][-1, :, -2] = 100000
    a = study.normalized_inputs(arrays, np.array([0, 1]))
    b = study.normalized_inputs(other, np.array([0, 1]))
    np.testing.assert_array_equal(a['summaries'][:2], b['summaries'][:2])
    np.testing.assert_array_equal(a['tokens'][:2], b['tokens'][:2])


def test_preregistered_eligibility_and_group_partitions_are_consistent():
    rows, _, _, _, plan = study.prepare()
    assert plan['provenance']['audit_sha256'] == study.digest(study.AUDIT)
    groups = {}
    recordings = {}
    for row in rows:
        assert row['target'] in row['candidates']
        assert row['current'] in row['candidates']
        assert row['target_onset'] > row['current_onset']
        assert all(e[0] <= row['current_onset'] for e in row['history'])
        assert row['recording'] not in plan['eligibility_exclusions']['flagged_recordings']
        groups.setdefault(row['group'], set()).add(row['split'])
        recordings.setdefault(row['recording'], set()).add(row['split'])
    assert all(len(s) == 1 for s in groups.values())
    assert all(len(s) == 1 for s in recordings.values())
    assert {r['split'] for r in rows} == {'train', 'validation', 'heldout'}
