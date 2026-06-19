"""Multi-axis evidence profile (migrated, proven scoring).

Six INDEPENDENT axes, each categorical + calibrated confidence. No single
label, and no axis claims validated semantics — that requires Track C biology.
Scorers take PRIMITIVE signals so the profile is unit-testable and decoupled
from how the signals were obtained. Pure stdlib.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class EvidenceProfile:
    """The full multi-axis verdict — a profile, not a single label."""

    species: str
    axes: dict[str, AxisResult] = field(default_factory=dict)

    @property
    def validated_semantics(self) -> bool:
        """Computationally we NEVER validate semantics — only Track C biology can."""
        sv = self.axes.get("semantic_validation")
        return sv is not None and sv.level == "playback-supported"

    @property
    def overall_confidence(self) -> float:
        if not self.axes:
            return 0.0
        return sum(a.confidence for a in self.axes.values()) / len(self.axes)

    def as_dict(self) -> dict:
        return {
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
    semantic_status: str = "none",
) -> EvidenceProfile:
    """Assemble all six axes from primitive signals."""
    p = EvidenceProfile(species=species)
    p.axes["sequence_recombination"] = score_sequence_recombination(lz_normalized)
    p.axes["functional_aliasing"] = score_functional_aliasing(alias_density, alias_pvalue)
    p.axes["call_unitization"] = score_call_unitization(n_hard, n_soft, n_total_boundaries)
    p.axes["context_binding"] = score_context_binding(n_context_bound)
    p.axes["turn_structure"] = score_turn_structure(n_turn_bigrams, entrained)
    p.axes["semantic_validation"] = score_semantic_validation(semantic_status)
    return p
