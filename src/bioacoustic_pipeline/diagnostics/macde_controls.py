"""Past-only occupancy and temporal-order controls for MACDE's frozen pilot."""
import numpy as np


def history_controls(history, seed):
    history = np.asarray(history)
    if history.shape != (52, 12) or not np.isfinite(history).all():
        raise ValueError('Requires exactly 52 supported past/current physical states')
    order = np.r_[np.random.default_rng(seed).permutation(50), 50, 51]
    return (np.r_[history.mean(0), history.std(0)], history.ravel(),
            history[order].ravel())
