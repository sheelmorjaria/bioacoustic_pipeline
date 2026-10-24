"""Mechanical label-support gate; no fitting, pooling, dropping or target scoring."""
from __future__ import annotations

from collections import Counter


def label_support(rows, *, min_fit=5, min_heldout=2, min_dates=3):
    """Conservative screening floors are not a statistical power calculation."""
    if len({r['event_id'] for r in rows}) != len(rows):
        raise ValueError('Event IDs must be unique')
    allowed = {'fit','heldout','unassigned'}
    for r in rows:
        if r['partition'] not in allowed:
            raise ValueError('Unknown partition')
        if not r['target'] or str(r['target']).strip() in ('NA','nan','None'):
            raise ValueError('Missing response-type target')
        if r['partition'] != 'unassigned' and r['date'] is None:
            raise ValueError('Assigned rows require date group')
    dates={p:{r['date'] for r in rows if r['partition']==p} for p in ('fit','heldout')}
    if dates['fit'] & dates['heldout']:
        raise ValueError('Shared date across fit and heldout')
    labels=sorted({r['target'] for r in rows})
    table={}
    for label in labels:
        entry={}
        for partition in ('fit','heldout','unassigned'):
            subset=[r for r in rows if r['target']==label and r['partition']==partition]
            entry[partition+'_count']=len(subset)
            entry[partition+'_dates']=len({r['date'] for r in subset if r['date'] is not None})
        table[label]=entry
    unseen=[label for label in labels if table[label]['heldout_count']>0 and table[label]['fit_count']==0]
    # Unassigned labels do not define a modelling class by themselves.
    assigned_labels=[label for label in labels if table[label]['fit_count']+table[label]['heldout_count']>0]
    sparse_fit=[label for label in assigned_labels if table[label]['fit_count']<min_fit]
    sparse_held=[label for label in assigned_labels if table[label]['heldout_count']<min_heldout]
    blocked=[]
    if len(assigned_labels)<2:
        blocked.append('fewer_than_two_response_classes')
    if unseen:
        blocked.append('unseen_heldout_target_class')
    if sparse_fit:
        blocked.append('insufficient_fit_class_support')
    if sparse_held:
        blocked.append('insufficient_heldout_class_support')
    if min(len(dates['fit']),len(dates['heldout']))<min_dates:
        blocked.append('insufficient_date_clusters')
    return {'fit_allowed':not blocked,'blocked_by':blocked,
            'partition_counts':dict(Counter(r['partition'] for r in rows)),
            'date_counts':{p:len(d) for p,d in dates.items()},
            'class_support':table,'heldout_labels_absent_from_fit':unseen,
            'sparse_fit_classes_below_5':sparse_fit,
            'sparse_heldout_classes_below_2':sparse_held,
            'screening_floors':{'fit_examples_per_class':min_fit,
                                'heldout_examples_per_class':min_heldout,
                                'dates_in_each_assigned_partition':min_dates},
            'screening_limits':'conservative support floors, not a statistical power guarantee; unseen fit class is a separate structural failure',
            'cohort_modified':False,'labels_pooled':False,'models_fitted':False}
