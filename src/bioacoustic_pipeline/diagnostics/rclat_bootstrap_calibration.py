"""Synthetic coverage calibration for the crossed positive-weight request bootstrap."""

import math

import numpy as np

from .rclat_communication import _joined, _probabilities

GRID = np.array([-1.5, -0.5, 0.5, 1.5])


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _predictions(eta, alternative):
    if alternative:
        return np.full_like(eta, 0.5), _sigmoid(0.75 * eta)
    return _sigmoid(eta + 0.4), _sigmoid(eta - 0.4)


def population_difference(scenario, *, nodes=80):
    """Population loss contrast integrates normal speaker+listener effects.

    Every speaker's covariate grid is balanced, including the unequal-size
    scenario, so this is the equal-stimulus, equal-speaker superpopulation
    estimand. Predictions are fixed; no fitting uncertainty is simulated.
    """
    x, w = np.polynomial.hermite.hermgauss(nodes)
    sd = math.hypot(scenario["speaker_sd"], scenario["listener_sd"])
    response = (_sigmoid(GRID[:, None] + math.sqrt(2) * sd * x[None, :]) @ w) / math.sqrt(math.pi)
    a, b = _predictions(GRID, scenario["alternative"])
    d = -response * np.log(a / b) - (1 - response) * np.log((1 - a) / (1 - b))
    answer = float(d.mean())
    return 0.0 if abs(answer) < 1e-14 else answer


def simulate(scenario, *, seed, speakers=18, listeners=82, min_raters=12, max_raters=31):
    if speakers < 2 or not 1 <= min_raters <= max_raters <= listeners:
        raise ValueError("Unsupported cluster/rater counts")
    if min(scenario["speaker_sd"], scenario["listener_sd"]) < 0:
        raise ValueError("Negative random-effect scale")
    rng = np.random.default_rng(seed)
    speaker_effect = rng.normal(0, scenario["speaker_sd"], speakers)
    listener_effect = rng.normal(0, scenario["listener_sd"], listeners)
    stimuli = []
    responses = []
    c1 = {}
    c2 = {}
    for s in range(speakers):
        repetitions = int(rng.integers(1, 3)) if scenario["uneven"] else 1
        for j, eta in enumerate(np.tile(GRID, repetitions)):
            uid = f"s{s:02d}_u{j:02d}"
            stimuli.append(
                {
                    "stimulus_id": uid,
                    "source_filename": uid,
                    "speaker_id": f"s{s:02d}",
                    "text_id": f"t{j % 4}",
                }
            )
            a, b = _predictions(np.array([eta]), scenario["alternative"])
            c1[uid] = float(a[0])
            c2[uid] = float(b[0])
            n = int(rng.integers(min_raters, max_raters + 1))
            observed = rng.choice(listeners, size=n, replace=False)
            probability = _sigmoid(eta + speaker_effect[s] + listener_effect[observed])
            values = rng.binomial(1, probability)
            responses.extend(
                {
                    "source_filename": uid,
                    "listener_id": f"l{int(listener):03d}",
                    "perceived_request": int(y),
                }
                for listener, y in zip(observed, values)
            )
    return {"stimuli": stimuli, "responses": responses, "c1": c1, "c2": c2}


def fast_crossed_interval(stimuli, responses, c1, c2, *, seed, replicates):
    """Dense batched equivalent of the reference kernel; preserves RNG draw order."""
    if not isinstance(replicates, int) or replicates < 2:
        raise ValueError("At least two bootstrap replicates required")
    joined = _joined(stimuli, responses)
    p1, p2 = _probabilities(stimuli, c1), _probabilities(stimuli, c2)
    uids = sorted(p1)
    listeners = sorted({listener for _, listener, _ in joined})
    speakers = sorted({s["speaker_id"] for s, _, _ in joined})
    ui = {u: i for i, u in enumerate(uids)}
    li = {listener: i for i, listener in enumerate(listeners)}
    si = {s: i for i, s in enumerate(speakers)}
    numerator = np.zeros((len(uids), len(listeners)))
    incidence = np.zeros_like(numerator)
    aggregation = np.zeros((len(uids), len(speakers)))
    for s, listener, y in joined:
        u = s["stimulus_id"]
        i, j = ui[u], li[listener]
        a, b = p1[u], p2[u]
        numerator[i, j] = -y * math.log(a / b) - (1 - y) * math.log((1 - a) / (1 - b))
        incidence[i, j] = 1
        aggregation[i, si[s["speaker_id"]]] = 1
    aggregation /= aggregation.sum(axis=0, keepdims=True)
    point = float(((numerator.sum(1) / incidence.sum(1)) @ aggregation).mean())
    weights = np.maximum(
        np.random.default_rng(seed).exponential(size=(replicates, len(listeners) + len(speakers))),
        np.finfo(float).tiny,
    )
    lw, sw = weights[:, : len(listeners)], weights[:, len(listeners) :]
    means = ((lw @ numerator.T) / (lw @ incidence.T)) @ aggregation
    draws = (means * sw).sum(1) / sw.sum(1)
    return {
        "difference": point,
        "interval95": np.quantile(draws, [0.025, 0.975], method="linear").tolist(),
        "seed": seed,
        "replicates": replicates,
        "speaker_clusters": len(speakers),
        "listener_clusters": len(listeners),
        "method": "paired crossed Exp(1) positive weights; vectorised reference-equivalent kernel",
    }


def wilson_interval(successes, total):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    scale = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [max(0.0, centre - radius), min(1.0, centre + radius)]
