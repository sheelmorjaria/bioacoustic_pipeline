"""Whole-day source partition and bounded target selection for a new TierA study."""
import numpy as np
import torch
from bioacoustic_pipeline.embeddings.macde import eligible_mask


def partition_sources(records):
    mapping = {('or60yw70', '092812'): 'fit', ('or60yw70', '093012'): 'validation',
               ('or60yw70', '100112'): 'heldout_day', ('gy6or6', '032612'): 'cross_bird'}
    result = {}
    for record in records:
        key = record['recording_id']
        if key in result:
            raise ValueError('Repeated source recording')
        source = (record['bird_id'], record['archive_day_code'])
        if source not in mapping:
            raise ValueError('Source outside prospective study population')
        result[key] = mapping[source]
    return result


def selected_positions(valid):
    valid = np.asarray(valid, bool)
    d = np.diff(np.r_[False, valid, False].astype(int))
    strict = np.zeros_like(valid)
    for start, end in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if end-start >= 128:
            strict[start:end] = True
    if len(valid) < 128:
        return np.array([], int)
    eligible = np.flatnonzero(eligible_mask(torch.from_numpy(strict)[None])[0].numpy())
    if len(eligible) < 16:
        return np.array([], int)
    return eligible[np.linspace(0, len(eligible)-1, 16, dtype=int)]
