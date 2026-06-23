"""Repertoire taxonomy — classify a corpus's vocal repertoire from validated diagnostics.

Architecture v1.0 Step 3. Pure stdlib (no data dependencies): given a ``RepertoireProfile`` of
already-validated per-corpus diagnostics — DUSTED motif-family stats (BENCHMARK §22) plus
individual-ID / context association Cramér's V (§18–§26) and optional social-lift (§9) — assign a
repertoire TYPE that explains WHY the corpus shows the structure it does. The type is a SYNTHESIS
label over validated numbers, not a new measurement.

Types:
  discrete_syllable        — diverse, strongly recurrent, GT-pure motifs (Bengalese finch song)
  graded                   — intermediate motif coverage, emitter-signatured bouts (bat)
  sparse_repetitive        — very low coverage, unstable motifs, one-unit repeats (infant marmoset)
  individually_stereotyped — atomic calls where caller identity dominates context (rook)
  socially_grounded        — context recovery needs social covariates (bat; a secondary flag)
  human_lexical            — human word-structure anchor (LibriSpeech; recurrent terms are NOT words)

Classification is calibrated to the §22 + binding numbers so each reference corpus lands on its
validated type; thresholds are documented inline. A corpus may carry SECONDARY flags
(identity_dominant / socially_grounded) alongside its primary type — a single label would lose the
bat's "graded + emitter-signatured + socially-grounded" richness.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass


class RepertoireType:
    """Repertoire type vocabulary (string constants — codebase idiom, not Enum)."""

    DISCRETE_SYLLABLE = "discrete_syllable"
    GRADED = "graded"
    SPARSE_REPETITIVE = "sparse_repetitive"
    INDIVIDUALLY_STEREOTYPED = "individually_stereotyped"
    SOCIALLY_GROUNDED = "socially_grounded"
    HUMAN_LEXICAL = "human_lexical"
    UNKNOWN = "unknown"


_DESCRIPTIONS = {
    RepertoireType.DISCRETE_SYLLABLE: (
        "Diverse inventory of stable, strongly recurrent, GT-pure motifs "
        "(e.g. Bengalese finch song). The only repertoire whose discovered terms are GT-validated."
    ),
    RepertoireType.GRADED: (
        "Intermediate motif coverage with graded/emitter-signatured structure; DTW over-merging "
        "risk (e.g. Egyptian fruit bat bouts)."
    ),
    RepertoireType.SPARSE_REPETITIVE: (
        "Very low motif coverage and unstable families; one-unit repeats dominate "
        "(e.g. infant marmoset calls)."
    ),
    RepertoireType.INDIVIDUALLY_STEREOTYPED: (
        "Atomic whole calls where caller identity dominates over context; calls are caller-bound "
        "(e.g. rook)."
    ),
    RepertoireType.SOCIALLY_GROUNDED: (
        "Context recovery needs social covariates — acoustics alone undercount function "
        "(e.g. bat; the §9 social-lift signal)."
    ),
    RepertoireType.HUMAN_LEXICAL: (
        "Human word-structure anchor. Cross-speaker word structure generalizes, but recurrent "
        "acoustic terms are NOT decoded words (semantic ceiling)."
    ),
    RepertoireType.UNKNOWN: "Insufficient diagnostics to classify.",
}


@dataclass
class RepertoireProfile:
    """Per-corpus validated diagnostics → a repertoire type.

    DUSTED fields mirror ``summarize_discovery`` (BENCHMARK §22, uniform thr=0.15 HAC).
    ``emitter_v`` / ``context_v`` are token↔label Cramér's V (individual-ID / context).
    ``social_lift`` is the D/A acoustic-vs-acoustic+social accuracy ratio (§9; bat only).
    Any field may be ``None`` when a corpus lacks that diagnostic (the classifier handles it).
    """

    corpus: str
    family_entropy: float | None = None  # H(norm) — near-max for all (weak discriminator)
    top_family_fraction: float | None = None
    null_ratio: float | None = None  # observed/null motif support (>1 ⇒ real recurrence)
    bootstrap_stability: float | None = None  # top-K Jaccard across resamples
    n_motif_families: int | None = None
    coverage: float | None = None
    emitter_v: float | None = None  # individual-ID association
    context_v: float | None = None  # context association
    social_lift: float | None = None  # D/A explicit-implicit (bat only)
    is_human: bool = False
    notes: str = ""
    source: str = ""  # BENCHMARK section citations

    @property
    def identity_dominant(self) -> bool:
        return (
            self.emitter_v is not None
            and self.emitter_v > 0.30
            and self.context_v is not None
            and self.emitter_v > self.context_v * 1.5
        )

    @property
    def socially_grounded(self) -> bool:
        return self.social_lift is not None and self.social_lift > 1.3

    @property
    def secondary_flags(self) -> list[str]:
        flags = []
        if self.identity_dominant:
            flags.append("identity_dominant")
        if self.socially_grounded:
            flags.append("socially_grounded")
        return flags

    @property
    def repertoire_type(self) -> str:
        return classify_repertoire(self)

    @property
    def description(self) -> str:
        return _DESCRIPTIONS.get(self.repertoire_type, "")

    def as_dict(self) -> dict:
        return {
            "corpus": self.corpus,
            "repertoire_type": self.repertoire_type,
            "secondary_flags": self.secondary_flags,
            "family_entropy": self.family_entropy,
            "top_family_fraction": self.top_family_fraction,
            "null_ratio": self.null_ratio,
            "bootstrap_stability": self.bootstrap_stability,
            "n_motif_families": self.n_motif_families,
            "coverage": self.coverage,
            "emitter_v": self.emitter_v,
            "context_v": self.context_v,
            "social_lift": self.social_lift,
            "description": self.description,
            "source": self.source,
            "notes": self.notes,
        }


def classify_repertoire(p: RepertoireProfile) -> str:
    """Assign a primary repertoire type. Calibrated to §22 + binding so each reference corpus
    lands on its validated type.

    Decision order (each threshold documented):
      1. human anchor → human_lexical
      2. has real motif structure (coverage ≥ 0.10):
           - strongly recurrent + stable (null>2.0 & bootstrap>0.45) → discrete_syllable (Bengalese)
           - else → graded (bat)
      3. no/low motif structure:
           - identity dominates (emitter_v>0.30 & emitter_v>1.5×context_v) → individually_stereotyped (rook)
           - very low coverage & unstable → sparse_repetitive (infant marmoset)
      4. else unknown
    """
    if p.is_human:
        return RepertoireType.HUMAN_LEXICAL
    has_motif = p.coverage is not None and p.coverage >= 0.10
    if has_motif:
        if (
            p.null_ratio is not None
            and p.null_ratio > 2.0
            and p.bootstrap_stability is not None
            and p.bootstrap_stability > 0.45
        ):
            return RepertoireType.DISCRETE_SYLLABLE
        return RepertoireType.GRADED
    # No/low motif coverage.
    if p.identity_dominant and (p.null_ratio is None or p.null_ratio < 2.0):
        return RepertoireType.INDIVIDUALLY_STEREOTYPED
    if (
        p.coverage is not None
        and p.coverage < 0.10
        and (p.bootstrap_stability is None or p.bootstrap_stability < 0.30)
    ):
        return RepertoireType.SPARSE_REPETITIVE
    if p.socially_grounded:
        return RepertoireType.SOCIALLY_GROUNDED
    return RepertoireType.UNKNOWN
