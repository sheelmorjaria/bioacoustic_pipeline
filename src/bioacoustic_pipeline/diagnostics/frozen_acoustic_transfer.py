"""Inference-only transforms: no target-domain statistics or optimisation."""
import numpy as np


def frozen_scale(x, mask, state):
    x, mask = np.asarray(x), np.asarray(mask, bool)
    mean, scale = np.asarray(state['mean']), np.asarray(state['scale'])
    support = np.asarray(state['support'], bool)
    if x.shape != mask.shape or x.shape[-1] != len(mean) or np.any(scale <= 0):
        raise ValueError('Frozen feature schema mismatch')
    valid = mask & support & np.isfinite(x)
    return np.where(valid, (x-mean)/scale, 0).astype(np.float32), valid


def frozen_ridge(state, x):
    x = np.asarray(x)
    if not np.isfinite(x).all() or x.shape[-1] != len(state['input_mean']):
        raise ValueError('Unsupported Ridge inputs')
    return (((x-state['input_mean'])/state['input_scale']) @ state['weight'] +
            state['intercept'])*state['output_scale'] + state['output_mean']
