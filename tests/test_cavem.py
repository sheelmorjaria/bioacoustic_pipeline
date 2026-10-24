# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""CAVEM (Contextual Acoustic and Vocal Event Model) — mechanics tests.

TDD suite written BEFORE the implementation. Tests assert MECHANICS
only (masking, causality, metadata separation, endpoint construction,
split integrity, baseline correctness); scientific numbers are produced
by the runners as diagnostics, never asserted here (repo convention).

Semantic-ceiling note baked into the design: nothing in CAVEM asserts
meaning; convergence of response modalities is EVIDENCE, never a
definition (the H0-Unified iff-clause is deliberately not encoded).
"""

from __future__ import annotations

import numpy as np
import pytest

from bioacoustic_pipeline.contextual.cavem import (
    DT_CAP_MS,
    EventRecord,
    build_response_endpoints,
    elapsed_features,
    masked_acoustic_targets,
    partition_by_group,
)


class TestEventRecordMetadataSeparation:
    def test_metadata_is_explicit_not_inferred(self):
        """caller/recipient/context are separate declared fields — the
        constructor must not accept an 'infer_from_audio' style path."""
        rec = EventRecord(
            acoustic=np.zeros((5, 12)),
            valid=np.ones(5, bool),
            dt_prev_ms=None,
            duration_ms=50.0,
            caller="A",
            recipient="B",
            context=None,
        )
        assert rec.caller == "A" and rec.recipient == "B" and rec.context is None

    def test_first_event_allows_none_dt_and_caller(self):
        rec = EventRecord(
            acoustic=np.zeros((5, 12)),
            valid=np.ones(5, bool),
            dt_prev_ms=None,
            duration_ms=50.0,
            caller=None,
            recipient=None,
            context=None,
        )
        assert rec.dt_prev_ms is None

    def test_invalid_rows_never_counted_as_content(self):
        rec = EventRecord(
            acoustic=np.zeros((5, 12)),
            valid=np.array([True, False, False, True, True]),
            dt_prev_ms=100.0,
            duration_ms=50.0,
            caller=None,
            recipient=None,
            context=None,
        )
        assert rec.n_valid == 3


class TestElapsedFeatures:
    def test_none_dt_maps_to_zero_vector_prefix(self):
        f = elapsed_features(None)
        assert f.shape == (2,)
        assert f[0] == 0.0 and f[1] == 0.0  # first event: no elapsed time

    def test_dt_is_log_scaled_and_capped(self):
        f_small = elapsed_features(100.0)
        f_big = elapsed_features(10_000_000.0)
        f_cap = elapsed_features(DT_CAP_MS)
        assert f_big[0] == f_cap[0]  # capped
        assert f_small[0] < f_cap[0]
        assert np.isfinite(f_small).all() and np.isfinite(f_big).all()


class TestResponseEndpoints:
    def test_addressee_response_within_window(self):
        """Y=1 iff a subsequent event by the RECIPIENT starts within W ms."""
        events = [
            EventRecord(np.zeros((3, 12)), np.ones(3, bool), None, 40.0, "A", "B", None),
            # B responds 500 ms later -> within 1000 ms window
            EventRecord(np.zeros((3, 12)), np.ones(3, bool), 500.0, 40.0, "B", "A", None),
            # A again 2000 ms later -> outside a 1000 ms window from event 0's end
            EventRecord(np.zeros((3, 12)), np.ones(3, bool), 2000.0, 40.0, "A", "C", None),
        ]
        y = build_response_endpoints(events, window_ms=1000.0, who="recipient")
        assert y == [1, 0, 0]

    def test_any_response_endpoint_ignores_identity(self):
        events = [
            EventRecord(np.zeros((3, 12)), np.ones(3, bool), None, 40.0, "A", "B", None),
            EventRecord(np.zeros((3, 12)), np.ones(3, bool), 300.0, 40.0, "C", "D", None),
        ]
        y = build_response_endpoints(events, window_ms=1000.0, who="any")
        assert y == [1, 0]

    def test_self_continuation_is_not_a_response(self):
        events = [
            EventRecord(np.zeros((3, 12)), np.ones(3, bool), None, 40.0, "A", "B", None),
            EventRecord(np.zeros((3, 12)), np.ones(3, bool), 200.0, 40.0, "A", "B", None),
        ]
        y = build_response_endpoints(events, window_ms=1000.0, who="recipient")
        assert y == [0, 0]  # caller's own next call never counts

    def test_window_measures_from_event_end(self):
        events = [
            EventRecord(np.zeros((3, 12)), np.ones(3, bool), None, 500.0, "A", "B", None),
            EventRecord(np.zeros((3, 12)), np.ones(3, bool), 1400.0, 40.0, "B", "A", None),
        ]
        # gap from END of event 0 to start of event 1 = 1400 - 500 = 900 ms
        y = build_response_endpoints(events, window_ms=1000.0, who="recipient")
        assert y == [1, 0]


class TestMaskedAcousticTargets:
    def test_invalid_positions_are_never_targets(self):
        valid = np.array([True, True, False, True, True])
        rng = np.random.RandomState(0)
        m = masked_acoustic_targets(valid, mask_frac=0.4, rng=rng)
        # masked set must be a subset of VALID positions
        assert set(m).issubset({i for i in range(5) if valid[i]})
        # unmasked valid positions remain observed
        assert set(range(5)) - set(m) >= {i for i in range(5) if not valid[i]}

    def test_mask_fraction_approximate(self):
        valid = np.ones(1000, bool)
        m = masked_acoustic_targets(valid, mask_frac=0.3, rng=np.random.RandomState(1))
        assert 0.2 < len(m) / 1000 < 0.4

    def test_distinguishes_missing_masked_and_observed(self):
        """output codes: 0=observed, 1=masked(target), 2=invalid(never target)."""
        valid = np.array([True, False, True, True])
        m = masked_acoustic_targets(valid, mask_frac=0.5, rng=np.random.RandomState(2))
        codes = np.zeros(4, int)
        codes[~valid] = 2
        codes[list(m)] = 1
        assert (codes == 2).sum() == 1
        assert (codes == 1).sum() == len(m)
        assert set(np.where(codes == 1)[0]).issubset(np.where(valid)[0])


class TestPartitionByGroup:
    def test_no_group_overlap_and_deterministic(self):
        groups = [f"g{i % 7}" for i in range(70)]
        p1 = partition_by_group(groups, train_frac=0.7, val_frac=0.5, seed=0)
        p2 = partition_by_group(groups, train_frac=0.7, val_frac=0.5, seed=0)
        assert p1 == p2  # deterministic
        fit = {g for g, v in p1.items() if v == "fit"}
        val = {g for g, v in p1.items() if v == "val"}
        ev = {g for g, v in p1.items() if v == "eval"}
        assert not (fit & val) and not (fit & ev) and not (val & ev)
        assert fit | val | ev == set(groups)

    def test_all_groups_assigned(self):
        groups = ["a", "b", "c", "d", "d"]
        p = partition_by_group(groups, train_frac=0.5, val_frac=0.5, seed=0)
        assert set(p) == {"a", "b", "c", "d"}


class TestSemanticCeilingFraming:
    def test_module_carries_no_meaning_definition(self):
        """The H0-Unified iff-convergence clause must NOT be encoded:
        the module defines response prediction as EVIDENCE, and exposes
        no function asserting meaning from convergence."""
        import bioacoustic_pipeline.contextual.cavem as mod

        forbidden = [n for n in dir(mod) if "meaning" in n.lower() or "semantic" in n.lower()]
        assert forbidden == []


class TestCausalContextModel:
    """Torch-dependent mechanics: causality and learnability on synthetic
    fixtures (skipped when torch is unavailable — core stays importable)."""

    def _fixture(self, n_seq=60, length=8, history_dependent=True, seed=0):
        """Synthetic event sequences. When history_dependent, Y_i depends on
        the PRECEDING event's acoustic direction — learnable only with a
        causal context model, not from the current event alone."""
        rng = np.random.RandomState(seed)
        seqs, ys = [], []
        for _ in range(n_seq):
            evs = []
            state = rng.rand()
            for i in range(length):
                dur = 30.0 + 10.0 * rng.rand()
                dt = 100.0 + 900.0 * rng.rand()
                mark = np.zeros((4, 12))
                mark[:, 0] = state  # latent 'direction' carried in feature 0
                evs.append(
                    EventRecord(
                        acoustic=mark + 0.01 * rng.randn(4, 12),
                        valid=np.ones(4, bool),
                        dt_prev_ms=None if i == 0 else dt,
                        duration_ms=dur,
                        caller="A" if state < 0.5 else "B",
                        recipient="B" if state < 0.5 else "A",
                        context=int(state < 0.5),
                    )
                )
                state = rng.rand()
            seqs.append(evs)
            if history_dependent:
                # Y_i = 1 iff previous event's mark was high: pure history signal
                ys.append([1 if evs[i - 1].acoustic[0, 0] > 0.5 else 0 for i in range(length)] if length else [])
            else:
                ys.append([int(rng.rand() < 0.5) for _ in range(length)])
        return seqs, ys

    def test_causal_states_do_not_leak_the_future(self):
        cavem = pytest.importorskip("bioacoustic_pipeline.contextual.cavem")
        CausalGRU = cavem.CausalGRU
        torch = pytest.importorskip("torch")
        seqs, _ = self._fixture(n_seq=4)
        model = CausalGRU(event_dim=12, hidden=16)
        base = model.causal_states(seqs[0])
        # perturb the LAST event; earlier states must be bit-identical
        evs2 = [type(e).model_validate(e.model_copy(update={})) if False else e for e in seqs[0]]
        evs2[-1] = EventRecord(
            acoustic=seqs[0][-1].acoustic + 1.0,
            valid=seqs[0][-1].valid,
            dt_prev_ms=seqs[0][-1].dt_prev_ms,
            duration_ms=seqs[0][-1].duration_ms,
            caller=seqs[0][-1].caller,
            recipient=seqs[0][-1].recipient,
            context=seqs[0][-1].context,
        )
        pert = model.causal_states(evs2)
        for i in range(len(base) - 1):
            assert torch.allclose(base[i], pert[i]), f"future leaked into state {i}"

    def test_gru_learns_history_signal_baseline_cannot(self):
        cavem = pytest.importorskip("bioacoustic_pipeline.contextual.cavem")
        pytest.importorskip("torch")
        seqs, ys = self._fixture(n_seq=80, history_dependent=True, seed=1)
        model = cavem.CausalGRU(event_dim=12, hidden=16)
        model.fit(seqs, ys, epochs=200, lr=5e-3, seed=0)
        acc = model.sequence_accuracy(seqs, ys)
        assert acc > 0.85, f"GRU failed to learn a pure history signal ({acc:.2f})"
        base_acc = cavem.current_event_baseline_accuracy(seqs, ys)
        assert base_acc < 0.75, "baseline suspiciously solves a history-only task (leak?)"

    def test_no_false_advantage_when_history_irrelevant(self):
        cavem = pytest.importorskip("bioacoustic_pipeline.contextual.cavem")
        seqs, ys = self._fixture(n_seq=80, history_dependent=False, seed=2)
        base_acc = cavem.current_event_baseline_accuracy(seqs, ys)
        assert 0.3 < base_acc < 0.75  # coin-flip task: no oracle advantage

    def test_incremental_delta_positive_exactly_when_history_helps(self):
        """REVISION NOTE: thresholds recalibrated when incremental_delta
        was corrected to out-of-sample scoring on BOTH arms (the original
        in-sample GRU accuracy rewarded memorisation — the NI control
        fixture caught it at ~0.95 accuracy on coin flips). With honest
        60/40 sequence splits the HD fixture yields ~+0.29 (vs baseline
        ~0.57) and NI ~0; thresholds 0.25 keep the test demanding while
        matching the corrected estimand."""
        cavem = pytest.importorskip("bioacoustic_pipeline.contextual.cavem")
        seqs_hd, ys_hd = self._fixture(n_seq=100, history_dependent=True, seed=3)
        seqs_ni, ys_ni = self._fixture(n_seq=100, history_dependent=False, seed=4)
        d_hd = cavem.incremental_delta(seqs_hd, ys_hd, epochs=300, seed=0)
        d_ni = cavem.incremental_delta(seqs_ni, ys_ni, epochs=300, seed=0)
        assert d_hd["delta"] > 0.25
        assert abs(d_ni["delta"]) < 0.25
