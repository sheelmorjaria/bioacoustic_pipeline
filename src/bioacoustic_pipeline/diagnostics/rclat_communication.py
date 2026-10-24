"""Listener-grounded request interpretation primitives; no automatic study fitting."""

from collections import defaultdict
import math

import numpy as np
import torch
from torch import nn


def canonicalize_responses(responses, text_aliases):
    """Apply only caller-declared metadata aliases; retain source IDs and judgments."""
    if any(
        not isinstance(k, str) or not isinstance(v, str) or not k or not v
        for k, v in text_aliases.items()
    ):
        raise ValueError("Explicit nonempty script alias strings required")
    result = []
    for row in responses:
        item = dict(row)
        if "text_id" in row:
            item["source_text_id"] = row["text_id"]
            item["text_id"] = text_aliases.get(row["text_id"], row["text_id"])
        result.append(item)
    return result


def _joined(stimuli, responses):
    names = {r["source_filename"]: r for r in stimuli}
    ids = {r["stimulus_id"] for r in stimuli}
    if not stimuli or len(names) != len(stimuli) or len(ids) != len(stimuli):
        raise ValueError("Empty or duplicate stimulus inventory")
    seen = set()
    joined = []
    for r in responses:
        if r["source_filename"] not in names:
            raise ValueError("unknown response stimulus")
        s = names[r["source_filename"]]
        if any(key in r and str(r[key]) != str(s[key]) for key in ["speaker_id", "text_id"]):
            raise ValueError("Response identity metadata disagrees with stimulus")
        listener = str(r.get("listener_id", "")).strip()
        if not listener or r.get("perceived_request") not in (0, 1):
            raise ValueError("Explicit listener identity and binary human response required")
        key = (s["stimulus_id"], listener)
        if key in seen:
            raise ValueError("duplicate listener/stimulus judgment")
        seen.add(key)
        joined.append((s, listener, int(r["perceived_request"])))
    if {s["stimulus_id"] for s, _, _ in joined} != ids:
        raise ValueError("Every stimulus requires actual listener judgments")
    return joined


def listener_targets(stimuli, responses):
    """Retain empirical distributions; production-intent fields are never read."""
    grouped = defaultdict(list)
    for s, listener, y in _joined(stimuli, responses):
        grouped[s["stimulus_id"]].append((listener, y, s))
    return {
        uid: {
            "speaker_id": rows[0][2]["speaker_id"],
            "text_id": rows[0][2]["text_id"],
            "request_probability": sum(y for _, y, _ in rows) / len(rows),
            "listeners": len(rows),
            "listener_ids": sorted(listener for listener, _, _ in rows),
        }
        for uid, rows in grouped.items()
    }


def fit_script_baseline(targets, training_ids, strength=1):
    if (
        not math.isfinite(strength)
        or strength <= 0
        or not training_ids
        or len(set(training_ids)) != len(training_ids)
    ):
        raise ValueError("Unique training stimulus IDs and positive fixed smoothing required")
    if any(uid not in targets for uid in training_ids):
        raise ValueError("Unknown training target")
    grouped = defaultdict(list)
    for uid in training_ids:
        row = targets[uid]
        grouped[row["text_id"]].append(row["request_probability"])
    grand = (sum(targets[u]["request_probability"] for u in training_ids) + strength * 0.5) / (
        len(training_ids) + strength
    )
    return {
        "scripts": {
            text: (sum(p) + strength * 0.5) / (len(p) + strength) for text, p in grouped.items()
        },
        "unseen_script_probability": grand,
        "training_ids": list(training_ids),
    }


def _probabilities(stimuli, probabilities):
    if set(probabilities) != {r["stimulus_id"] for r in stimuli}:
        raise ValueError("Predictions must cover exactly the evaluated stimuli")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities.values()):
        raise ValueError("Predictions must be finite probabilities")
    return {uid: min(1 - 1e-6, max(1e-6, float(p))) for uid, p in probabilities.items()}


def _weighted_loss(joined, probabilities, listener_weights=None, speaker_weights=None):
    weighted = defaultdict(lambda: [0.0, 0.0])
    speakers = {}
    for s, listener, y in joined:
        uid = s["stimulus_id"]
        p = probabilities[uid]
        weight = 1 if listener_weights is None else listener_weights[listener]
        weighted[uid][0] += weight * (-y * math.log(p) - (1 - y) * math.log1p(-p))
        weighted[uid][1] += weight
        speakers[uid] = s["speaker_id"]
    groups = defaultdict(list)
    for uid, (total, weight) in weighted.items():
        groups[speakers[uid]].append(total / weight)
    ws = {s: 1 if speaker_weights is None else speaker_weights[s] for s in groups}
    return sum(ws[s] * float(np.mean(v)) for s, v in groups.items()) / sum(ws.values())


def speaker_macro_loss(stimuli, responses, probabilities):
    """Equal stimulus weights within speakers; listener response counts do not dominate."""
    return _weighted_loss(_joined(stimuli, responses), _probabilities(stimuli, probabilities))


def crossed_interval(stimuli, responses, c1, c2, *, seed, replicates):
    """Paired crossed positive-weight bootstrap, conditional on fixed predictions.

    A single Exp(1) weight per listener is shared across all that listener's
    judgments; a separate Exp(1) speaker weight is shared across its stimuli.
    Positive weights retain every response and stimulus. This is not a frozen
    scientific decision rule: a study must prospectively fix its use and draws.
    """
    if not isinstance(replicates, int) or replicates < 2:
        raise ValueError("At least two bootstrap replicates required")
    joined = _joined(stimuli, responses)
    a, b = _probabilities(stimuli, c1), _probabilities(stimuli, c2)
    listeners = sorted({listener for _, listener, _ in joined})
    speakers = sorted({s["speaker_id"] for s, _, _ in joined})
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        lw = dict(
            zip(listeners, np.maximum(rng.exponential(size=len(listeners)), np.finfo(float).tiny))
        )
        sw = dict(
            zip(speakers, np.maximum(rng.exponential(size=len(speakers)), np.finfo(float).tiny))
        )
        draws.append(_weighted_loss(joined, a, lw, sw) - _weighted_loss(joined, b, lw, sw))
    return {
        "difference": _weighted_loss(joined, a) - _weighted_loss(joined, b),
        "interval95": np.quantile(draws, [0.025, 0.975], method="linear").tolist(),
        "seed": seed,
        "replicates": replicates,
        "speaker_clusters": len(speakers),
        "listener_clusters": len(listeners),
        "method": "paired crossed Exp(1) positive weights; equal-stimulus speaker-macro loss",
    }


def readiness(*, wording, source_use, methodology_frozen):
    flags = {
        "independent_wording": wording.get("ready") is True,
        "source_use": source_use is True,
        "methodology_frozen": methodology_frozen is True,
    }
    blockers = [name for name, value in flags.items() if not value]
    return {"ready": not blockers, "blockers": blockers, "automatic_training_authorized": False}


class TemporalRequestModel(nn.Module):
    """Capacity-matched C1/C2: identical model, constant-summary versus fine sequence.

    Only verified lexical tensors and acoustic measurements enter the model.
    No listener identity, production condition, filenames or target distribution.
    Acoustic values must use training-fitted scalers; scientific feature choices
    and fixed resampling grids require a subsequent protocol freeze.
    """

    def __init__(self, lexical_dim, acoustic_dim, steps=32, hidden=16):
        super().__init__()
        if min(lexical_dim, acoustic_dim, steps, hidden) <= 0:
            raise ValueError("Positive dimensions required")
        self.lexical_dim = lexical_dim
        self.acoustic_dim = acoustic_dim
        self.steps = steps
        self.temporal = nn.GRU(acoustic_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lexical_dim + acoustic_dim + hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, lexical, summary, trajectory, *, arm):
        batch = len(lexical)
        if (
            lexical.shape != (batch, self.lexical_dim)
            or summary.shape != (batch, self.acoustic_dim)
            or trajectory.shape != (batch, self.steps, self.acoustic_dim)
        ):
            raise ValueError("Fixed feature schema required; no time-varying mask inputs")
        if arm == "C1":
            sequence = summary[:, None, :].expand(-1, self.steps, -1)
        elif arm == "C2":
            sequence = trajectory
        else:
            raise ValueError("Temporal comparison supports C1/C2 only")
        _, hidden = self.temporal(sequence)
        return self.head(torch.cat([lexical, summary, hidden[-1]], dim=-1)).squeeze(-1)
