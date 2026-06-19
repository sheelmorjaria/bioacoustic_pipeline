"""Falsification report — every claim ships with how it could be wrong.

For each axis asserting positive structure, emit the hypothesis plus the
specific tests that would refute it: null (permutation p), replication
(held-out split), confound (speaker/session/device), or biology (Track C
playback). The 'biology' track is the only route to validated meaning.

Migrated proven content. Pure stdlib.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from bioacoustic_pipeline.verdict_engine.evidence_profile import EvidenceProfile

_UNIVERSAL_SEMANTICS = (
    "(universal) no acoustic token stream establishes meaning without Track C biology"
)

# Per-axis refutation tests: (label, track) where track in {null, replication,
# confound, biology} — must match verdict_engine.schema.FALSIFICATION_TRACKS.
_AXIS_TESTS: dict[str, list[tuple[str, str]]] = {
    "sequence_recombination": [
        ("LZ complexity does not exceed within-file-shuffle null (p >= 0.05)", "null"),
        ("LZ collapses under colony/speaker-held-out replication", "replication"),
        ("complexity explained by recording-device or session identity", "confound"),
    ],
    "functional_aliasing": [
        ("alias density does not exceed within-file-shuffle null (p >= 0.05)", "null"),
        ("alias groups dissolve under cross-file (global-unigram) null", "null"),
        ("alias edges unstable under bootstrap (survival < 0.5)", "null"),
        ("alias structure explained by speaker/session, not syntactic role", "confound"),
        ("playback alias substitution yields no receiver-equivalent response", "biology"),
    ],
    "call_unitization": [
        ("hard/soft mix flips under NBD threshold recalibration", "null"),
        ("mix unstable across recording sessions/devices", "confound"),
        ("expert-annotated unit boundaries disagree with NBD boundaries", "biology"),
    ],
    "context_binding": [
        ("context-bound n-grams do not exceed shuffled-context null", "null"),
        ("patterns confined to one recording session/device", "confound"),
    ],
    "turn_structure": [
        ("turn bigrams explained by overlap/co-occurrence chance, not syntax", "null"),
        ("rhythmic entrainment disappears in held-out dyads/colonies", "replication"),
    ],
    "semantic_validation": [
        (_UNIVERSAL_SEMANTICS, "biology"),
    ],
}

_POSITIVE_LEVELS = {
    "sequence_recombination": {"medium", "high"},
    "functional_aliasing": {"weak", "strong"},
    "call_unitization": {"segmented", "mixed", "holistic"},
    "context_binding": {"weak", "strong"},
    "turn_structure": {"rhythmic", "constrained"},
    "semantic_validation": {"none", "behavioral", "playback-supported"},
}


def makes_positive_claim(axis_name: str, level: str) -> bool:
    return level in _POSITIVE_LEVELS.get(axis_name, set())


def falsification_tests(profile: EvidenceProfile) -> list[dict]:
    """Structured list of {axis, level, confidence, makes_positive_claim, tests}."""
    out = []
    for name, axis in profile.axes.items():
        tests = _AXIS_TESTS.get(name, [])
        out.append({
            "axis": name,
            "level": axis.level,
            "confidence": round(axis.confidence, 3),
            "makes_positive_claim": makes_positive_claim(name, axis.level),
            "falsification_tests": [{"label": lbl, "track": tr} for lbl, tr in tests],
        })
    return out


def format_falsification_report(profile: EvidenceProfile) -> str:
    lines = ["=" * 64, "FALSIFICATION REPORT", "=" * 64]
    lines.append(f"Species: {profile.species}")
    lines.append(
        "Framing: the pipeline does NOT infer meaning. It decomposes vocal audio "
        "into\nsegments, representations, sequence roles, and an evidence profile "
        "that generates\nfalsifiable hypotheses for behavioral validation."
    )
    lines.append("")
    for name, axis in profile.axes.items():
        lines.append(f"--- {name.upper()} : {axis.level} (conf {axis.confidence:.0%}) ---")
        if makes_positive_claim(name, axis.level):
            lines.append(f"  Hypothesis: {name} = {axis.level} ({axis.evidence})")
            lines.append("  Would be falsified by:")
            for lbl, tr in _AXIS_TESTS.get(name, []):
                lines.append(f"    [{tr}] {lbl}")
        else:
            lines.append(f"  No positive claim (level = {axis.level}). {axis.evidence}")
        lines.append("")
    sv = profile.axes.get("semantic_validation")
    if sv and sv.level != "playback-supported":
        lines.append("--- UNIVERSAL SEMANTICS CAVEAT ---")
        lines.append("  No axis establishes biological MEANING. Only Track C")
        lines.append("  (playback / receiver-response / alias-substitution) can, and it")
        lines.append("  has not been performed.")
    lines.append("=" * 64)
    return "\n".join(lines)
