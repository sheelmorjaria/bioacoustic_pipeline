"""Multi-axis evidence profile (migrated, proven scoring).

Six INDEPENDENT axes, each categorical + calibrated confidence. No single
label, and no axis claims validated semantics — that requires Track C biology.
Scorers take PRIMITIVE signals so the profile is unit-testable and decoupled
from how the signals were obtained. Pure stdlib.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bioacoustic_pipeline.diagnostics.sequence_sufficiency import (
    NOT_EVALUABLE,
    SufficiencyResult,
)


@dataclass
class AxisResult:
    """One axis of the evidence profile."""

    name: str
    level: str
    confidence: float  # 0-1
    evidence: str
    ci: tuple[float, float] | None = None

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


class AbstentionReason:
    """Canonical reasons an axis abstains (Architecture v1.0 freeze).

    Abstention is a scientific OUTPUT, not a failure: an axis that does not measure is
    never a biological null. ``not_evaluable`` = could not compute (sparse data);
    ``not_applicable`` = refused to interpret (wrong domain, e.g. domain gate). This
    vocabulary names the structured reason behind each abstention so the report can
    surface a full trail. String constants (not Enum) to match the codebase idiom
    (NOT_EVALUABLE / NOT_APPLICABLE). Mirrors ``abstention_reasons`` in
    configs/segmentation_router.yaml.
    """

    TOO_FEW_CALLS = "too_few_calls"
    LOW_PARENT_EVENT_CONFIDENCE = "low_parent_event_confidence"
    UNSTABLE_MOTIF_BOOTSTRAP = "unstable_motif_bootstrap"
    NULL_NOT_EXCEEDED = "null_not_exceeded"
    DOMAIN_REJECTED = "domain_rejected"
    CONTEXT_LABELS_INSUFFICIENT = "context_labels_insufficient"
    IDENTITY_CONFOUND_UNRESOLVED = "identity_confound_unresolved"
    REGIME_UNCLASSIFIED = "regime_unclassified"


@dataclass
class EvidenceProfile:
    """The full multi-axis verdict — a profile, not a single label."""

    species: str
    axes: dict[str, AxisResult] = field(default_factory=dict)
    domain: dict | None = None  # domain-gate block (None for an un-gated communication run)
    # Architecture v1.0: abstention trail. Each (axis, reason) records WHY an axis did
    # not measure (reason = an AbstentionReason.* constant) — surfaced in as_dict/summary.
    # Abstention is a first-class scientific output, never a biological null.
    abstentions: list[tuple[str, str]] = field(default_factory=list)

    def record_abstention(self, axis: str, reason: str) -> None:
        """Record why an axis abstained (``reason`` = an AbstentionReason.* constant)."""
        self.abstentions.append((axis, reason))

    @property
    def abstention_reasons(self) -> dict[str, str]:
        return dict(self.abstentions)

    def is_abstained(self, axis: str) -> bool:
        return axis in self.abstention_reasons

    @property
    def validated_semantics(self) -> bool:
        """Computationally we NEVER validate semantics — only Track C biology can."""
        sv = self.axes.get("semantic_validation")
        return sv is not None and sv.level == "playback-supported"

    @property
    def overall_confidence(self) -> float:
        # not_applicable axes (domain-gate refusals) are abstentions, not measurements —
        # exclude them. Other axes (incl. not_evaluable) keep their confidence.
        scored = [a for a in self.axes.values() if a.level != "not_applicable"]
        if not scored:
            return 0.0
        return sum(a.confidence for a in scored) / len(scored)

    def as_dict(self) -> dict:
        out = {
            "species": self.species,
            "overall_confidence": round(self.overall_confidence, 3),
            "validated_semantics": self.validated_semantics,
            "axes": {
                k: {
                    "level": v.level,
                    "confidence": round(v.confidence, 3),
                    "evidence": v.evidence,
                }
                for k, v in self.axes.items()
            },
        }
        if self.abstentions:
            out["abstentions"] = [{"axis": ax, "reason": r} for ax, r in self.abstentions]
        if self.domain is not None:
            out["domain"] = self.domain
        return out

    def summary(self) -> str:
        lines = [
            f"Evidence profile [{self.species}] (overall conf {self.overall_confidence:.0%}, "
            f"semantics: {'UNVALIDATED' if not self.validated_semantics else 'validated'})"
        ]
        for name, axis in self.axes.items():
            ci = f" CI[{axis.ci[0]:.2f},{axis.ci[1]:.2f}]" if axis.ci else ""
            lines.append(f"  {name:24s} {axis.level:16s} conf={axis.confidence:.0%}{ci}")
        return "\n".join(lines)


def _distance_confidence(value: float, boundaries: tuple[float, ...], scale: float = 0.15) -> float:
    """At a boundary -> 0.5 (maximally uncertain); far from all -> 1.0."""
    d = min(abs(value - b) for b in boundaries)
    return min(1.0, 0.5 + (d / scale) * 0.5)


def _not_evaluable_axis(name: str, reason_detail: str, reason_code: str | None = None) -> AxisResult:
    """An abstaining axis: the diagnostic could not be run, NOT a biological null."""
    code = f" [{reason_code}]" if reason_code else ""
    return AxisResult(
        name, NOT_EVALUABLE, 0.0,
        f"NOT EVALUABLE{code}: {reason_detail or 'sequence input too sparse to test'}",
    )


NOT_APPLICABLE = "not_applicable"


def _not_applicable_axis(name: str, reason_detail: str) -> AxisResult:
    """A domain-gate refusal: the axis is not interpretable for this corpus's domain.

    Distinct from not_evaluable: not_evaluable = "could not compute (sparse data)";
    not_applicable = "refuse to interpret (wrong domain — e.g. environmental sound,
    not animal communication)". Never a biological null or a claim.
    """
    return AxisResult(
        name, NOT_APPLICABLE, 0.0,
        f"NOT APPLICABLE: {reason_detail}",
    )


def score_sequence_recombination(lz_normalized: float) -> AxisResult:
    if lz_normalized < 0.30:
        level = "low"
    elif lz_normalized < 0.60:
        level = "medium"
    else:
        level = "high"
    return AxisResult(
        "sequence_recombination", level,
        _distance_confidence(lz_normalized, (0.30, 0.60)),
        f"LZ complexity {lz_normalized:.3f} (thresholds 0.30/0.60)",
    )


def score_functional_aliasing(alias_density: float, alias_pvalue: float | None = None) -> AxisResult:
    if alias_pvalue is not None and alias_pvalue >= 0.05:
        return AxisResult(
            "functional_aliasing", "absent",
            min(1.0, 0.5 + (alias_pvalue - 0.05) * 2),
            f"alias density {alias_density:.1%} NOT significant "
            f"(p={alias_pvalue:.3f} >= 0.05) — consistent with chance",
        )
    level = "weak" if alias_density < 0.05 else "strong"
    if alias_pvalue is None:
        conf, note = 0.35, " (no null test run — provisional)"
    else:
        conf, note = min(1.0, 1.0 - alias_pvalue), f" (p={alias_pvalue:.3f})"
    return AxisResult(
        "functional_aliasing", level, conf,
        f"alias density {alias_density:.1%} -> {level}{note}",
    )


def score_call_unitization(n_hard: int, n_soft: int, n_total_boundaries: int) -> AxisResult:
    total = max(n_hard + n_soft, 1)
    frac_soft = n_soft / total
    level = "holistic" if frac_soft < 0.25 else ("segmented" if frac_soft > 0.60 else "mixed")
    return AxisResult(
        "call_unitization", level,
        _distance_confidence(frac_soft, (0.25, 0.60), scale=0.18),
        f"{n_hard} hard / {n_soft} soft boundaries (frac_soft={frac_soft:.2f})",
    )


def score_context_binding(n_context_bound: int) -> AxisResult:
    if n_context_bound == 0:
        level, conf = "absent", 0.9
    elif n_context_bound <= 3:
        level, conf = "weak", 0.6
    else:
        level, conf = "strong", 0.8
    return AxisResult(
        "context_binding", level, conf,
        f"{n_context_bound} context-bound sequence patterns",
    )


def score_turn_structure(n_turn_bigrams: int, entrained: bool | None = None) -> AxisResult:
    if n_turn_bigrams == 0:
        return AxisResult(
            "turn_structure", "absent", 0.7,
            "no cross-emitter turn bigrams (single-emitter or corpus N/A)",
        )
    if entrained is True:
        level, conf = "rhythmic", 0.75
    elif entrained is False:
        level, conf = "constrained", 0.6
    else:
        level, conf = "rhythmic", 0.5
    return AxisResult(
        "turn_structure", level, conf,
        f"{n_turn_bigrams} turn bigrams (entrainment="
        f"{'yes' if entrained else 'no' if entrained is False else 'unknown'})",
    )


def score_acoustic_coherence(token_coherence: float | None = None,
                             alias_divergence: float | None = None) -> AxisResult:
    """Acoustic coherence — do learned tokens map to coherent acoustic classes?

    An ACOUSTIC AUDIT, not a semantic claim: high within-token fingerprint
    similarity means VQ tokens are real acoustic classes (not latent artifacts).
    Driven by the acoustic-fingerprint channel; ``None`` (fingerprint off) -> absent.
    """
    if token_coherence is None:
        return AxisResult(
            "acoustic_coherence", "absent", 0.9,
            "no acoustic-fingerprint signal (enable run_pipeline(fingerprint=True))",
        )
    level = "weak" if token_coherence < 0.5 else "strong"
    div_note = f", alias divergence {alias_divergence:.2f}" if alias_divergence is not None else ""
    return AxisResult(
        "acoustic_coherence", level,
        _distance_confidence(token_coherence, (0.5,), scale=0.15),
        f"token-fingerprint coherence {token_coherence:.2f}{div_note}",
    )


def score_motif_structure(n_motifs: int | None = None, motif_null_p: float | None = None) -> AxisResult:
    """Motif structure — do recurrent sequence patterns exceed chance?

    An ACOUSTIC AUDIT of sequence regularity: motifs are recurring token/alias/acoustic
    patterns that survive null-model testing. NOT syntax — statistically tested
    sequence hypotheses. ``None`` (no motif mining) -> absent.
    """
    if n_motifs is None or n_motifs == 0:
        return AxisResult(
            "motif_structure", "absent", 0.9,
            "no motif mining performed (sequences too short or tokenizer unavailable)",
        )
    level = "weak" if n_motifs < 5 else "strong"
    p_note = f" (null p={motif_null_p:.3f})" if motif_null_p is not None else ""
    conf = 0.6 + min(0.3, n_motifs * 0.02)
    if motif_null_p is not None and motif_null_p < 0.05:
        conf = min(1.0, conf + 0.15)
    return AxisResult(
        "motif_structure", level, conf,
        f"{n_motifs} significant motifs{p_note}",
    )


def score_semantic_validation(status: str = "none") -> AxisResult:
    """Hard-coded: computation alone can never establish meaning."""
    if status not in {"none", "behavioral", "playback-supported"}:
        status = "none"
    return AxisResult(
        "semantic_validation", status, 1.0,
        "semantics require Track C (playback/receiver-response) biology; "
        "not inferable from acoustic tokens",
    )


def compute_evidence_profile(
    species: str,
    *,
    lz_normalized: float,
    alias_density: float = 0.0,
    alias_pvalue: float | None = None,
    n_hard: int = 0,
    n_soft: int = 0,
    n_total_boundaries: int = 0,
    n_context_bound: int = 0,
    n_turn_bigrams: int = 0,
    entrained: bool | None = None,
    token_coherence: float | None = None,
    alias_acoustic_divergence: float | None = None,
    n_motifs: int | None = None,
    motif_null_p: float | None = None,
    semantic_status: str = "none",
    sufficiency: SufficiencyResult | None = None,
    context_evaluable: bool = True,
    context_reason: str | None = None,
    turn_evaluable: bool = True,
    turn_reason: str | None = None,
    domain_gate=None,
) -> EvidenceProfile:
    """Assemble all eight axes from primitive signals.

    Sequence-dependent axes (sequence_recombination, functional_aliasing,
    motif_structure) ABSTAIN to ``not_evaluable`` when ``sufficiency`` says the
    token sequences are too sparse to test — never a false biological null.
    context_binding / turn_structure abstain when their own ``*_evaluable`` flag
    is False (e.g. no annotation labels / no multi-emitter turn data).
    """
    p = EvidenceProfile(species=species)
    if sufficiency is not None and not sufficiency.evaluable:
        for ax_name in ("sequence_recombination", "functional_aliasing", "motif_structure"):
            p.axes[ax_name] = _not_evaluable_axis(
                ax_name, sufficiency.reason_detail, sufficiency.reason)
    else:
        p.axes["sequence_recombination"] = score_sequence_recombination(lz_normalized)
        p.axes["functional_aliasing"] = score_functional_aliasing(alias_density, alias_pvalue)
        p.axes["motif_structure"] = score_motif_structure(n_motifs, motif_null_p)
    p.axes["call_unitization"] = score_call_unitization(n_hard, n_soft, n_total_boundaries)
    if context_evaluable:
        p.axes["context_binding"] = score_context_binding(n_context_bound)
    else:
        p.axes["context_binding"] = _not_evaluable_axis(
            "context_binding", context_reason or "no annotation labels available", "no_annotation_labels")
    if turn_evaluable:
        p.axes["turn_structure"] = score_turn_structure(n_turn_bigrams, entrained)
    else:
        p.axes["turn_structure"] = _not_evaluable_axis(
            "turn_structure", turn_reason or "no multi-emitter turn data", "no_turn_data")
    p.axes["acoustic_coherence"] = score_acoustic_coherence(token_coherence, alias_acoustic_divergence)
    p.axes["semantic_validation"] = score_semantic_validation(semantic_status)

    # Domain gate: if the corpus is rejected for bioacoustic-communication inference
    # (e.g. ESC-50 environmental sound), REFUSE the ethological-interpretation axes —
    # not_applicable, not a biological null. Acoustic/structural axes stay scored.
    if domain_gate is not None and getattr(domain_gate, "rejected", False):
        reason = getattr(domain_gate, "reason_detail", "domain gate rejected communication inference")
        for ax in getattr(domain_gate, "disabled_axes", ()):
            if ax in p.axes:
                p.axes[ax] = _not_applicable_axis(ax, reason)
        p.domain = {
            "dataset": getattr(domain_gate, "dataset", ""),
            "dataset_role": getattr(domain_gate, "dataset_role", ""),
            "verdict": getattr(domain_gate, "dataset_verdict", ""),
            "communication_status": getattr(domain_gate, "communication_status", ""),
            "clip_verdict": getattr(domain_gate, "verdict", ""),
            "source_type": getattr(domain_gate, "source_type", ""),
            "reason": list(getattr(domain_gate, "reason", []) or []),
        }
    return p
