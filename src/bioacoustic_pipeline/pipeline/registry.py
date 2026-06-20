"""DiagnosticRegistry — plug-in diagnostics, not a giant run_pipeline.

Diagnostics are registered ``fn(events, ctx) -> dict`` and run in registration order.
``sequence_sufficiency`` runs first and stashes its :class:`SufficiencyResult` on ``ctx``;
diagnostics with ``requires_sufficiency`` abstain to ``not_evaluable`` when the gate fails;
diagnostics whose axis is in the domain gate's ``disabled_axes`` abstain to
``not_applicable``; diagnostics needing absent annotations (``requires_context`` /
``requires_emitter``) abstain to ``skipped``. This makes abstention principled and keeps
``run_pipeline`` from becoming a monolithic script.

Phase 1 registers sufficiency + LZ + motif. Phase 2 adds alias/context/coherence by
registering more fns — no orchestrator change.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from bioacoustic_pipeline.pipeline.contracts import (
    STATUS_COMPUTED,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_EVALUABLE,
    STATUS_SKIPPED,
    DiagnosticResult,
)


@dataclass
class _Spec:
    name: str
    fn: Callable
    axis: str | None = None
    requires_sufficiency: bool = False
    requires_context: bool = False
    requires_emitter: bool = False
    gated_by_domain: bool = True


class DiagnosticRegistry:
    """Ordered registry of diagnostics with domain/sufficiency/data gating."""

    def __init__(self) -> None:
        self._specs: list[_Spec] = []
        self._by_name: dict[str, _Spec] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        *,
        axis: str | None = None,
        requires_sufficiency: bool = False,
        requires_context: bool = False,
        requires_emitter: bool = False,
        gated_by_domain: bool = True,
        first: bool = False,
    ) -> "DiagnosticRegistry":
        spec = _Spec(
            name,
            fn,
            axis,
            requires_sufficiency,
            requires_context,
            requires_emitter,
            gated_by_domain,
        )
        if first:
            self._specs.insert(0, spec)
        else:
            self._specs.append(spec)
        self._by_name[name] = spec
        return self

    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    def run(self, events: list, ctx: dict) -> dict[str, DiagnosticResult]:
        gate = ctx["domain_gate"]
        out: dict[str, DiagnosticResult] = {}
        for spec in self._specs:
            # Domain gate: axis refused for this corpus -> not_applicable.
            if spec.gated_by_domain and spec.axis and spec.axis in gate.disabled_axes:
                out[spec.name] = DiagnosticResult(
                    spec.name,
                    STATUS_NOT_APPLICABLE,
                    axis=spec.axis,
                    reason=f"domain gate disabled axis {spec.axis!r}",
                )
                continue
            # Data availability gates.
            if spec.requires_context and not getattr(gate, "has_context", False):
                out[spec.name] = DiagnosticResult(
                    spec.name, STATUS_SKIPPED, axis=spec.axis, reason="no context annotations"
                )
                continue
            if spec.requires_emitter and not getattr(gate, "has_emitter", False):
                out[spec.name] = DiagnosticResult(
                    spec.name, STATUS_SKIPPED, axis=spec.axis, reason="no emitter annotations"
                )
                continue
            # Sequence-sufficiency gate (sufficiency itself is exempt — it sets the gate).
            if spec.requires_sufficiency:
                suff = ctx.get("sufficiency")
                if suff is None or not suff.evaluable:
                    out[spec.name] = DiagnosticResult(
                        spec.name,
                        STATUS_NOT_EVALUABLE,
                        axis=spec.axis,
                        reason="sequence sufficiency gate abstained",
                    )
                    continue
            try:
                value = spec.fn(events, ctx)
                out[spec.name] = DiagnosticResult(
                    spec.name, STATUS_COMPUTED, value=value, axis=spec.axis
                )
            except Exception as exc:  # noqa: BLE001 — one diagnostic failing must not abort the run
                out[spec.name] = DiagnosticResult(
                    spec.name, STATUS_SKIPPED, axis=spec.axis, reason=f"error: {exc}"
                )
        return out


# --------------------------------------------------------------------------- #
# Phase-1 diagnostic functions (adapters: read primitives off ctx, which the
# orchestrator derived from the DualStreamSegment stream).
# --------------------------------------------------------------------------- #


def _sufficiency_fn(events, ctx):
    from bioacoustic_pipeline.diagnostics.sequence_sufficiency import evaluate_sufficiency  # noqa: PLC0415

    suff = evaluate_sufficiency(ctx["sequences"])
    ctx["sufficiency"] = suff  # stash for downstream requires_sufficiency diagnostics
    return suff.as_dict()


def _lempel_ziv_fn(events, ctx):
    from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity  # noqa: PLC0415

    flat = [t for s in ctx["sequences"] for t in s][:5000]
    return lempel_ziv_complexity(flat).as_dict()


def _motif_fn(events, ctx):
    from bioacoustic_pipeline.diagnostics import motif_miner as mm  # noqa: PLC0415

    seqs = ctx["sequences"]
    seed = ctx.get("seed", 0)
    if sum(len(s) for s in seqs) < 10:
        return {"n_motifs": 0}
    raw = mm.mine_ngrams(seqs, min_n=2, max_n=5, min_support=3)
    motifs = mm.build_motifs(raw, seqs, motif_type="token")
    try:
        mp = round(float(mm.motif_support_null(seqs, n_permutations=100, seed=seed).p_value), 4)
    except Exception:  # noqa: BLE001
        mp = None
    return {
        "n_motifs": len(motifs),
        "motif_null_p": mp,
        "top_motifs": [m.as_dict() for m in motifs[:5]],
    }


def default_registry() -> DiagnosticRegistry:
    """Phase-1 registry: sufficiency (first) + LZ + motif."""
    r = DiagnosticRegistry()
    r.register("sequence_sufficiency", _sufficiency_fn, first=True)
    r.register(
        "lempel_ziv", _lempel_ziv_fn, axis="sequence_recombination", requires_sufficiency=True
    )
    r.register("motif", _motif_fn, axis="motif_structure", requires_sufficiency=True)
    return r
