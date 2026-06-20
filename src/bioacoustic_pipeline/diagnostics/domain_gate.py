"""Domain gate — refuse ethological-communication inference on out-of-domain corpora.

Runs BEFORE any biological/communication diagnostic. An environmental-sound benchmark
(e.g. BEANS-Zero ESC-50: 2000 5-s environmental clips, 50 classes) is NOT an annotated
animal-communication corpus, so the pipeline must REFUSE communication claims on it — even
though some ESC-50 labels (dog, rooster, crickets, chirping_birds, …) are biological
sources. The gate therefore separates two independent questions:

  * clip SOURCE  — biological sound vs non-biological sound
  * communication-CORPUS suitability — bioacoustic-communication candidate vs not

The desired negative-control result is NOT "the model finds no structure" (environmental
sounds have plenty); it is *"the pipeline detects acoustic structure but refuses to infer
biological communication."* The gate marks the ethological-interpretation axes
``not_applicable`` while the acoustic/structural audits (NBD, fingerprint, tokens,
acoustic_coherence, call_unitization, sequence_recombination) still run.

Three-class communication taxonomy (clip level):

  NON_BIOLOGICAL_SOUND                          — rain, engine, siren, …
  BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK — dog, rooster, crickets, …
  BIOACOUSTIC_COMMUNICATION_CANDIDATE          — a real annotated communication corpus

Dataset level: ESC-50 ⇒ NON_BIOACOUSTIC_COMMUNICATION (rejected, useful only as OOD /
environmental negative control). Pure stdlib.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bioacoustic_pipeline.verdict_engine.schema import AXIS_NAMES

# --- clip-level communication verdict taxonomy ---
NON_BIOLOGICAL_SOUND = "NON_BIOLOGICAL_SOUND"
BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK = "BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK"
BIOACOUSTIC_COMMUNICATION_CANDIDATE = "BIOACOUSTIC_COMMUNICATION_CANDIDATE"
# dataset-level
NON_BIOACOUSTIC_COMMUNICATION = "NON_BIOACOUSTIC_COMMUNICATION"

# --- source types ---
BIOLOGICAL_SOURCE = "BIOLOGICAL_SOURCE"
NON_BIOLOGICAL_SOURCE = "NON_BIOLOGICAL_SOURCE"
AMBIGUOUS_SOURCE = "AMBIGUOUS_SOURCE"

REJECTED = "REJECTED_FOR_BIOACOUSTIC_COMMUNICATION"
CANDIDATE = "CANDIDATE"

# ESC-50 labels whose source is biological (animal/insect) — still a NON-COMMUNICATION benchmark.
BIOLOGICAL_ESC50_LABELS = frozenset(
    {
        "dog",
        "rooster",
        "pig",
        "cow",
        "frog",
        "cat",
        "hen",
        "insects",
        "sheep",
        "crow",
        "crickets",
        "chirping_birds",
    }
)

# dataset id -> corpus role
DATASET_ROLES = {
    "esc-50": "negative_control_environmental_sound",
    "beans_zero:esc-50": "negative_control_environmental_sound",
}
DEFAULT_ROLE = "bioacoustic_communication"

# Axes still computed (acoustic/structural audits) when the gate rejects communication.
APPLICABLE_AXES = frozenset({"sequence_recombination", "call_unitization", "acoustic_coherence"})
# Ethological-interpretation axes REFUSED (→ not_applicable) when the gate rejects.
DISABLED_AXES = frozenset(
    {
        "functional_aliasing",
        "motif_structure",
        "turn_structure",
        "context_binding",
        "semantic_validation",
    }
)

_ESC50_REJECTION_REASONS = [
    "ESC-50 is an environmental sound-classification dataset, not annotated animal communication",
    "clip has no call-type timestamp sequence annotations",
    "clip has no emitter/addressee social metadata",
    "clip has no behavioral-context sequence annotations",
    "no evidence of communicative bout structure",
]


def _normalize(label: str | None) -> str:
    if not label:
        return ""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def classify_source(label: str | None) -> str:
    """Clip source type from an ESC-50 label (biological vs non-biological vs ambiguous)."""
    if not label:
        return AMBIGUOUS_SOURCE
    return (
        BIOLOGICAL_SOURCE if _normalize(label) in BIOLOGICAL_ESC50_LABELS else NON_BIOLOGICAL_SOURCE
    )


@dataclass
class DomainGateResult:
    """One domain-gate decision (per clip or per dataset)."""

    dataset: str
    dataset_role: str
    source_type: str
    communication_status: str  # REJECTED | CANDIDATE
    verdict: str  # clip-level communication verdict
    dataset_verdict: str  # dataset-level communication verdict
    reason: list[str] = field(default_factory=list)
    disabled_axes: set[str] = field(default_factory=set)
    applicable_axes: set[str] = field(default_factory=set)
    confidence: float = 0.99

    @property
    def rejected(self) -> bool:
        return self.communication_status == REJECTED

    @property
    def reason_detail(self) -> str:
        return "; ".join(self.reason) if self.reason else "domain gate rejected"

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "dataset_role": self.dataset_role,
            "source_type": self.source_type,
            "communication_status": self.communication_status,
            "verdict": self.verdict,
            "dataset_verdict": self.dataset_verdict,
            "reason": list(self.reason),
            "disabled_axes": sorted(self.disabled_axes),
            "applicable_axes": sorted(self.applicable_axes),
            "confidence": round(self.confidence, 3),
        }


def _resolve_role(dataset: str) -> str:
    if dataset in DATASET_ROLES:
        return DATASET_ROLES[dataset]
    if f"beans_zero:{dataset}" in DATASET_ROLES:
        return DATASET_ROLES[f"beans_zero:{dataset}"]
    return DEFAULT_ROLE


def gate_clip(dataset: str, label: str | None = None) -> DomainGateResult:
    """Domain-gate decision for one clip given its dataset (+ optional ESC-50 label)."""
    role = _resolve_role(dataset)
    if role == "negative_control_environmental_sound":
        src = classify_source(label)
        verdict = (
            BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK
            if src == BIOLOGICAL_SOURCE
            else NON_BIOLOGICAL_SOUND
        )
        return DomainGateResult(
            dataset=dataset,
            dataset_role=role,
            source_type=src,
            communication_status=REJECTED,
            verdict=verdict,
            dataset_verdict=NON_BIOACOUSTIC_COMMUNICATION,
            reason=list(_ESC50_REJECTION_REASONS),
            disabled_axes=set(DISABLED_AXES),
            applicable_axes=set(APPLICABLE_AXES),
            confidence=0.99,
        )
    # default: a real annotated communication corpus -> candidate, no axes disabled.
    return DomainGateResult(
        dataset=dataset,
        dataset_role=role,
        source_type=AMBIGUOUS_SOURCE,
        communication_status=CANDIDATE,
        verdict=BIOACOUSTIC_COMMUNICATION_CANDIDATE,
        dataset_verdict=BIOACOUSTIC_COMMUNICATION_CANDIDATE,
        reason=["corpus treated as a bioacoustic-communication candidate"],
        disabled_axes=set(),
        applicable_axes=set(AXIS_NAMES),
        confidence=0.5,
    )


def gate_dataset(dataset: str) -> DomainGateResult:
    """Corpus-level gate decision (no per-clip label)."""
    return gate_clip(dataset, label=None)


def summarize_dataset(results: list[DomainGateResult]) -> dict:
    """Aggregate per-clip gate decisions into the dataset-level summary block."""
    n = len(results)
    n_nonbio = sum(r.verdict == NON_BIOLOGICAL_SOUND for r in results)
    n_bio = sum(r.verdict == BIOLOGICAL_SOUND_NON_COMMUNICATION_BENCHMARK for r in results)
    n_cand = sum(r.communication_status == CANDIDATE for r in results)
    return {
        "dataset_verdict": (
            BIOACOUSTIC_COMMUNICATION_CANDIDATE if n_cand > 0 else NON_BIOACOUSTIC_COMMUNICATION
        ),
        "n_files": n,
        "n_nonbiological_source": n_nonbio,
        "n_biological_source_noncommunication": n_bio,
        "communication_candidate_files": n_cand,
    }
