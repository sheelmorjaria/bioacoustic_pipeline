"""H0-CMC evidence tiers — the causal-multimodal-convergence evidence ladder.

Architecture v1.0 Step 5 (the spine that connects every module to the verdict engine). The
H0-CMC framework defines a 6-rung evidence ladder (T0→T5) from purely acoustic structure to
ecologically validated function, and maps each rung to the verdict engine's
``semantic_validation`` axis. This module defines the tiers + the evidence dataclass; the
explicit-implicit test (Step 7) and social axis (Step 7) are companion modules.

The pipeline operates AUTONOMOUSLY at T0–T1 (acoustic + context structure). It GENERATES T2–T4
manifests (playback counterfactuals via the DSP Audio Splicer) for wet-lab validation. It can
INGEST T3–T5 results when they return (receiver response data fed back into the engine).

Tiers:
  T0 ACOUSTIC        — recurrent acoustic structure found (DUSTED/alternatives; §22/§33/§34)
  T1 CONTEXT         — context-associated cue (token↔context binding; §18/§23)
  T2 PLAYBACK_1M     — receiver detects the distinction (1 modality; the DSP splicer generates stimuli)
  T3 CONVERGENCE     — multimodal convergence: candidate functional interpretation
  T4 COUNTERFACTUAL  — causally supported: receiver response disrupted by the counterfactual
  T5 FIELD_REPLICATION — ecologically validated function (field fitness-relevant behavior)

Pure stdlib. NOT validated meaning until T3+ — encoded in both ``schema.validate_report`` and
``evidence_profile.score_semantic_validation``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class EvidenceTier(IntEnum):
    """The H0-CMC evidence ladder (T0→T5)."""

    T0_ACOUSTIC = 0  # recurrent acoustic structure (DUSTED/alternatives)
    T1_CONTEXT = 1  # context-associated cue (binding)
    T2_PLAYBACK_1M = 2  # receiver detects distinction (1 modality — playback stimuli generated)
    T3_CONVERGENCE = 3  # multimodal convergence → candidate functional interpretation
    T4_COUNTERFACTUAL = 4  # causally supported (counterfactual disrupts response)
    T5_FIELD_REPLICATION = 5  # ecologically validated function


# Maps a tier to the verdict engine's semantic_validation axis level.
_TIER_TO_SEMANTIC = {
    EvidenceTier.T0_ACOUSTIC: "none",
    EvidenceTier.T1_CONTEXT: "none",  # context-tagged but NOT meaning
    EvidenceTier.T2_PLAYBACK_1M: "none",  # playback-tagged but response not yet causal
    EvidenceTier.T3_CONVERGENCE: "candidate_functional_interpretation",
    EvidenceTier.T4_COUNTERFACTUAL: "causally_supported_signal",
    EvidenceTier.T5_FIELD_REPLICATION: "ecologically_validated_function",
}

_TIER_DESCRIPTIONS = {
    EvidenceTier.T0_ACOUSTIC: "Recurrent acoustic structure discovered (DUSTED / alternatives). "
    "Structure exists; no functional claim.",
    EvidenceTier.T1_CONTEXT: "Acoustic structure is context-associated (token↔context binding "
    "significant). Correlation, not causation.",
    EvidenceTier.T2_PLAYBACK_1M: "Playback stimuli generated; receiver response PENDING. "
    "The DSP Audio Splicer produces the counterfactual set; wet lab tests the distinction.",
    EvidenceTier.T3_CONVERGENCE: "Multimodal convergence: acoustic + behavioral + physiological "
    "responses align → candidate functional interpretation (NOT validated meaning yet).",
    EvidenceTier.T4_COUNTERFACTUAL: "Counterfactual playback disrupts receiver response → the "
    "signal is causally supported (the structure matters for communication).",
    EvidenceTier.T5_FIELD_REPLICATION: "Ecologically validated: the signal predicts "
    "fitness-relevant behavior in the field. The strongest computational evidence possible.",
}


@dataclass
class H0CMCEvidence:
    """One corpus's position on the H0-CMC evidence ladder.

    Fields at T0–T1 are computable autonomously. T2+ fields are ``None`` until wet-lab data
    returns. The ``semantic_validation`` mapping ties this to the verdict engine's axis.
    """

    corpus: str
    tier: EvidenceTier = EvidenceTier.T0_ACOUSTIC

    # T0–T1 (autonomous): acoustic + context structure scores
    acoustic_score: float = 0.0  # e.g. DUSTED null-ratio, RQA DET, motif purity
    context_score: float = 0.0  # token↔context Cramér's V (§18/§23)
    social_lift: float | None = None  # D/A from explicit-implicit test (§9; bat only)

    # T2–T4 (wet-lab, initially None): receiver response evidence
    behavioral_response: float | None = None  # R_behavioral (0–1; proportion responding)
    vocal_response: float | None = None  # R_vocal (change in receiver vocalisation rate)
    physiological_response: float | None = None  # R_physiological (e.g. heart-rate change)
    causal_specificity: float | None = (
        None  # I_causal: does the counterfactual specifically disrupt?
    )

    # T5 (field, None until ecology data)
    field_fitness_correlation: float | None = None

    notes: str = ""
    evidence_sources: list[str] = field(default_factory=list)  # BENCHMARK section citations

    @property
    def semantic_validation(self) -> str:
        """Maps to the verdict engine's semantic_validation axis level."""
        return _TIER_TO_SEMANTIC[self.tier]

    @property
    def description(self) -> str:
        return _TIER_DESCRIPTIONS[self.tier]

    @property
    def is_autonomous(self) -> bool:
        """T0–T1: the pipeline can establish this tier computationally."""
        return self.tier <= EvidenceTier.T1_CONTEXT

    @property
    def requires_wet_lab(self) -> bool:
        """T2+: requires playback / receiver-response biology."""
        return self.tier >= EvidenceTier.T2_PLAYBACK_1M

    def evidence_score(self) -> float:
        """Weighted evidence across the ladder (0–1). T0–T1 are computable; T2+ add wet-lab weight."""
        s = self.acoustic_score * 0.3 + self.context_score * 0.2
        if self.behavioral_response is not None:
            s += self.behavioral_response * 0.2
        if self.causal_specificity is not None:
            s += self.causal_specificity * 0.2
        if self.field_fitness_correlation is not None:
            s += self.field_fitness_correlation * 0.1
        return min(1.0, s)

    def as_dict(self) -> dict:
        return {
            "corpus": self.corpus,
            "tier": f"T{self.tier.value}",
            "tier_name": self.tier.name,
            "semantic_validation": self.semantic_validation,
            "description": self.description,
            "acoustic_score": self.acoustic_score,
            "context_score": self.context_score,
            "social_lift": self.social_lift,
            "behavioral_response": self.behavioral_response,
            "vocal_response": self.vocal_response,
            "physiological_response": self.physiological_response,
            "causal_specificity": self.causal_specificity,
            "field_fitness_correlation": self.field_fitness_correlation,
            "evidence_score": round(self.evidence_score(), 3),
            "is_autonomous": self.is_autonomous,
            "requires_wet_lab": self.requires_wet_lab,
            "notes": self.notes,
            "evidence_sources": self.evidence_sources,
        }


def classify_tier(
    acoustic_score: float,
    context_score: float,
    social_lift: float | None = None,
    behavioral_response: float | None = None,
    causal_specificity: float | None = None,
    field_fitness: float | None = None,
) -> EvidenceTier:
    """Assign the highest tier the evidence supports.

    The pipeline reaches T0 if there's ANY acoustic structure, T1 if context-associated.
    T2+ require wet-lab fields (behavioral/causal/field). This is the autonomous classifier —
    it never claims above T1 without biological data.
    """
    if field_fitness is not None and field_fitness > 0.3:
        return EvidenceTier.T5_FIELD_REPLICATION
    if causal_specificity is not None and causal_specificity > 0.5:
        return EvidenceTier.T4_COUNTERFACTUAL
    if behavioral_response is not None and behavioral_response > 0.3:
        return EvidenceTier.T3_CONVERGENCE
    if context_score > 0.15:
        return EvidenceTier.T1_CONTEXT
    if acoustic_score > 0.0:
        return EvidenceTier.T0_ACOUSTIC
    return EvidenceTier.T0_ACOUSTIC
