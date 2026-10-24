"""Fixed-model speaker-paired development summaries; no model-fitting inference."""

import math

import numpy as np


def paired_speaker_summary(ridge, gru, *, seed, replicates=10000):
    """Paired percentile bootstrap of equal-speaker losses, conditional on fit/sample.

    Does not cover model seeds, origin sampling, source selection or tuning.
    Positive differences favour GRU; a strictly positive lower bound is the
    development-only utility criterion. This is not communication crossed inference.
    """
    if (
        not ridge
        or set(ridge) != set(gru)
        or len(ridge) < 2
        or type(replicates) is not int
        or replicates < 2
    ):
        raise ValueError("At least two matched speaker losses and two draws required")
    speakers = sorted(ridge)
    r = np.array([ridge[s] for s in speakers], dtype=float)
    g = np.array([gru[s] for s in speakers], dtype=float)
    if not np.isfinite(r).all() or not np.isfinite(g).all() or (r < 0).any() or (g < 0).any():
        raise ValueError("Finite nonnegative paired losses required")
    d = r - g
    samples = np.random.default_rng(seed).integers(0, len(d), size=(replicates, len(d)))
    interval = np.quantile(d[samples].mean(1), [0.025, 0.975], method="linear").tolist()
    return dict(
        difference=float(d.mean()),
        interval95=interval,
        speakers=len(d),
        speaker_differences=dict(zip(speakers, d.tolist())),
        ridge_loss=float(r.mean()),
        gru_loss=float(g.mean()),
        seed=seed,
        replicates=replicates,
        development_utility_passed=bool(d.mean() > 0 and interval[0] > 0),
        scope="Conditional fixed-model fixed-origin development speaker resampling",
    )


def select_minimum(ordered_scores):
    scores = list(ordered_scores)
    if not scores or any(not math.isfinite(float(score)) or score < 0 for _, score in scores):
        raise ValueError("Nonempty finite nonnegative ordered selection scores required")
    return min(enumerate(scores), key=lambda item: (item[1][1], item[0]))[1][0]
