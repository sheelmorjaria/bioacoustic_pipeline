# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""CAVEM — Contextual Acoustic and Vocal Event Model (V1 mechanics).

Implements the advisor-directed contextual-modelling framework (2026-10-20)
at the mechanics level:

  Level A  acoustic encoder inputs (validity-supported trajectories);
  Level B  event-sequence machinery: EventRecord with EXPLICIT metadata
           (caller/recipient/context are declared fields, never inferred
           from audio), elapsed-time features, causal context states;
  Level C  response-endpoint construction (behavioural/vocal/physiological
           heads are separate by design; agreement across modalities is
           EVIDENCE, never a definition of meaning — the H0-Unified
           iff-clause is deliberately NOT encoded here: no function in
           this module asserts meaning, semantics, or convergence-defined
           interpretation).

Masking discipline follows the repo invariant: invalid/padded positions
are never prediction targets and are distinguished from masked-but-valid
positions. Splitting follows the recording-group discipline.

The real-data runner selects the self-supervised acoustic branch when a
corpus cannot reconstruct genuine event sequences (isolated recordings
must not be manufactured into conversations).

Heavy deps (torch, sklearn) are imported lazily; the core is importable
without them.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DT_CAP_MS = 60_000.0  # elapsed-time features cap at one minute


@dataclass
class EventRecord:
    """One vocal event with EXPLICIT metadata fields.

    ``caller``/``recipient``/``context`` are independently established
    annotations carried as metadata — never inferred from the acoustic
    array, and never required (None allowed).
    """

    acoustic: np.ndarray  # (T, F) trajectory
    valid: np.ndarray  # (T,) validity mask
    dt_prev_ms: Optional[float]  # gap from previous event's END (None for first)
    duration_ms: float
    caller: Optional[str] = None
    recipient: Optional[str] = None
    context: Optional[int] = None
    provenance: dict = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        return int(np.asarray(self.valid).sum())


def elapsed_features(dt_prev_ms: Optional[float]) -> np.ndarray:
    """Bounded (2,) encoding of elapsed time; first event -> zeros."""
    if dt_prev_ms is None:
        return np.zeros(2)
    d = min(max(float(dt_prev_ms), 0.0), DT_CAP_MS)
    return np.array([math.log1p(d) / math.log1p(DT_CAP_MS), d / DT_CAP_MS])


def build_response_endpoints(events, window_ms: float, who: str = "recipient"):
    """Y_i = 1 iff a subsequent event consistent with ``who`` starts
    within ``window_ms`` of event i's END.

    who='recipient': the responder must be event i's annotated recipient
    (a receiver-vocal-response observable). who='any': any subsequent
    event. The caller's own continuation never counts as a response.
    dt_prev_ms is start-to-start; end-gap = dt_prev[j] - duration[i] +
    intermediate spans.
    """
    n = len(events)
    y = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if who == "recipient" and events[i].recipient is None:
                continue
            if who == "recipient" and events[j].caller != events[i].recipient:
                continue
            if who == "any" and events[j].caller == events[i].caller:
                continue
            # start of j relative to end of i
            gap = 0.0
            for k in range(i + 1, j + 1):
                gap += events[k].dt_prev_ms or 0.0
                if k < j:
                    gap += events[k].duration_ms
            gap -= events[i].duration_ms
            if 0 <= gap <= window_ms:
                y[i] = 1
                break
    return y


def masked_acoustic_targets(valid, mask_frac: float, rng) -> list[int]:
    """Indices to mask: a subset of VALID positions only.

    Invalid positions are never targets (they stay excluded by the
    caller's loss masking); masked-but-valid positions become targets.
    """
    v = np.asarray(valid, bool)
    cand = np.where(v)[0]
    n_mask = int(round(mask_frac * len(cand)))
    if n_mask <= 0 or len(cand) == 0:
        return []
    return sorted(int(x) for x in rng.choice(cand, size=min(n_mask, len(cand)), replace=False))


def partition_by_group(groups, train_frac: float = 0.7, val_frac: float = 0.5, seed: int = 0):
    """Deterministic recording-group -> fit/val/eval partition (no group
    ever spans splits); val_frac is the share OF THE FIT SIDE used for
    stopping/model selection."""
    uniq = sorted(set(groups))
    part = {}
    fit_side = []
    for g in uniq:
        h = int(hashlib.sha1(f"{seed}|{g}".encode()).hexdigest()[:8], 16) % 100
        if h < 100 * train_frac:
            fit_side.append(g)
        else:
            part[g] = "eval"
    rng = np.random.RandomState(seed)
    fl = sorted(fit_side)
    rng.shuffle(fl)
    n_val = max(1, int(len(fl) * val_frac)) if fl else 0
    for g in fl[:n_val]:
        part[g] = "val"
    for g in fl[n_val:]:
        part[g] = "fit"
    return part


def event_input(e: EventRecord) -> np.ndarray:
    """Per-event model input: [acoustic mean over valid frames (F),
    elapsed (2), log duration (1)]."""
    a = np.asarray(e.acoustic, float)
    v = np.asarray(e.valid, bool)
    mu = a[v].mean(0) if v.any() else np.zeros(a.shape[1])
    return np.concatenate([mu, elapsed_features(e.dt_prev_ms), [math.log1p(e.duration_ms)]]).astype(
        np.float32
    )


def current_event_baseline_accuracy(seqs, ys, seed: int = 0) -> float:
    """Logistic baseline on CURRENT-event acoustics only, sequence-level
    split (fit on the first 60% of sequences, score the rest). This is
    the 'no conversational context' comparator."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y, seq_id = [], [], []
    for si, (evs, yy) in enumerate(zip(seqs, ys)):
        for e, t in zip(evs, yy):
            X.append(event_input(e)[:12])
            y.append(int(t))
            seq_id.append(si)
    if len(set(y)) < 2:
        return float("nan")
    X, y, seq_id = np.array(X), np.array(y), np.array(seq_id)
    seqs_uniq = sorted(set(seq_id.tolist()))
    cut = seqs_uniq[int(0.6 * len(seqs_uniq))]
    m_tr, m_ev = seq_id <= cut, seq_id > cut
    if m_ev.sum() == 0 or m_tr.sum() == 0 or len(set(y[m_tr])) < 2:
        return float("nan")
    sc = StandardScaler().fit(X[m_tr])
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(sc.transform(X[m_tr]), y[m_tr])
    return float((clf.predict(sc.transform(X[m_ev])) == y[m_ev]).mean())


try:  # torch-gated causal context model (core stays importable without it)
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    class CausalGRU:  # noqa: D101
        """Small causal event-context encoder (Level B).

        causal_states(events)[i] depends on E_0..E_i ONLY (information-
        availability discipline: forecasts never see future observations;
        bidirectional encoders would be retrospective-only and are not
        provided here by design).
        """

        def __init__(self, event_dim: int = 12, hidden: int = 32):
            self.event_dim = event_dim
            self.hidden = hidden
            self.gru = nn.GRU(input_size=event_dim + 3, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, 2)
            self._sc = None
            self._mu = None
            self._sd = None

        def _inputs(self, events):
            X = np.stack([event_input(e) for e in events])
            return X

        def _normalize_fit(self, seqs):
            allx = np.vstack([self._inputs(evs) for evs in seqs])
            self._mu = allx.mean(0)
            self._sd = allx.std(0) + 1e-6

        def causal_states(self, events):
            X = self._inputs(events)
            if self._mu is not None:
                X = (X - self._mu) / self._sd
            # unfitted model: NO normalization — any whole-sequence
            # normalization would couple early states to late events and
            # violate the causality contract
            with torch.no_grad():
                out, _ = self.gru(torch.tensor(X, dtype=torch.float32).unsqueeze(0))
            return [out[0, i] for i in range(len(events))]

        def fit(self, seqs, ys, epochs: int = 100, lr: float = 1e-3, seed: int = 0):
            torch.manual_seed(seed)
            self._normalize_fit(seqs)
            opt = torch.optim.Adam(
                list(self.gru.parameters()) + list(self.head.parameters()), lr=lr
            )
            lossf = nn.CrossEntropyLoss()
            data = []
            for evs, yy in zip(seqs, ys):
                X = (self._inputs(evs) - self._mu) / self._sd
                data.append(
                    (torch.tensor(X, dtype=torch.float32), torch.tensor(yy, dtype=torch.long))
                )
            self.gru.train()
            for _ in range(epochs):
                opt.zero_grad()
                loss = 0.0
                for X, y in data:
                    out, _ = self.gru(X.unsqueeze(0))
                    loss = loss + lossf(self.head(out[0]), y)
                (loss / len(data)).backward()
                opt.step()
            self.gru.eval()
            return self

        def sequence_accuracy(self, seqs, ys) -> float:
            correct = total = 0
            self.gru.eval()
            with torch.no_grad():
                for evs, yy in zip(seqs, ys):
                    X = (self._inputs(evs) - self._mu) / self._sd
                    out, _ = self.gru(torch.tensor(X, dtype=torch.float32).unsqueeze(0))
                    pred = self.head(out[0]).argmax(1).numpy()
                    correct += int((pred == np.array(yy)).sum())
                    total += len(yy)
            return correct / max(total, 1)

    def incremental_delta(seqs, ys, epochs: int = 200, seed: int = 0) -> dict:
        """Δ_response = acc(context model) − acc(current-event baseline).

        The advisor's primary model-selection criterion: incremental
        prediction of the observable endpoint, not embedding quality.
        BOTH arms are scored out-of-sample under the same sequence-level
        60/40 split (in-sample accuracy would reward memorisation and
        fake an incremental advantage — caught by the history-independent
        control fixture).
        """
        n = len(seqs)
        cut = int(0.6 * n)
        tr_s, tr_y = seqs[:cut], ys[:cut]
        ev_s, ev_y = seqs[cut:], ys[cut:]
        model = CausalGRU(event_dim=12, hidden=32).fit(tr_s, tr_y, epochs=epochs, seed=seed)
        acc = model.sequence_accuracy(ev_s, ev_y)
        base = current_event_baseline_accuracy(seqs, ys, seed=seed)
        return {
            "context_model_accuracy": round(float(acc), 4),
            "current_event_baseline": round(float(base), 4) if base == base else None,
            "delta": round(float(acc - base), 4) if base == base else None,
            "split": f"sequence-level 60/40 (fit {cut} / eval {n - cut})",
        }

except ImportError:  # pragma: no cover - torch-free environments
    CausalGRU = None  # type: ignore[assignment,misc]

    def incremental_delta(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("torch required for CausalGRU")
