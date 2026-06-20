"""Canonical verdict output schema — the contract every upstream module feeds.

Built FIRST (test-driven): the final "Evidence Profile + Falsification Report"
JSON shape is defined here before the modules that populate it. Upstream stages
(segmentation, embeddings, diagnostics) are stitched to satisfy this schema.

The validator enforces the ethological invariant: computation can NEVER assert
validated semantics — `validated_semantics` may be True only if a Track C
(biology) evidence entry is attached. A pure-computational report with
`validated_semantics: true` fails validation.

Pure stdlib. No model deps.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

AXIS_NAMES = (
    "sequence_recombination",
    "functional_aliasing",
    "call_unitization",
    "context_binding",
    "turn_structure",
    "acoustic_coherence",
    "motif_structure",
    "semantic_validation",
)

# ``not_evaluable``: a sequence/annotation-dependent axis ABSTAINS when its input
# is too sparse to test (see diagnostics.sequence_sufficiency). It is never a
# biological null — it states the diagnostic could not be run, so a reviewer
# cannot read it as "the species lacks this structure". Acoustic coherence and
# call unitization are not gated this way (per-segment signals, not sequences).
#
# ``not_applicable``: the domain gate (diagnostics.domain_gate) REFUSES an
# ethological-interpretation axis on an out-of-domain corpus (e.g. ESC-50
# environmental sound). Distinct from not_evaluable: not_evaluable = "could not
# compute (sparse data)"; not_applicable = "refuse to interpret (wrong domain)".
# Both are abstentions, never claims. Only the 5 ethological axes carry it.
AXIS_LEVELS = {
    "sequence_recombination": {"not_evaluable", "low", "medium", "high"},
    "functional_aliasing": {"not_evaluable", "not_applicable", "absent", "weak", "strong"},
    "call_unitization": {"holistic", "segmented", "mixed"},
    "context_binding": {"not_evaluable", "not_applicable", "absent", "weak", "strong"},
    "turn_structure": {"not_evaluable", "not_applicable", "absent", "rhythmic", "constrained"},
    "acoustic_coherence": {"absent", "weak", "strong"},
    "motif_structure": {"not_evaluable", "not_applicable", "absent", "weak", "strong"},
    "semantic_validation": {"none", "behavioral", "playback-supported", "not_applicable"},
}

FALSIFICATION_TRACKS = {"null", "replication", "confound", "biology"}

SCHEMA_VERSION = "1.0"


def validate_report(report: dict) -> list[str]:
    """Return a list of schema violations (empty == valid)."""
    errors: list[str] = []

    for key in ("schema_version", "species", "validated_semantics", "axes", "falsification"):
        if key not in report:
            errors.append(f"missing top-level key: {key!r}")

    if errors:
        return errors

    if report["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    axes = report["axes"]
    if set(axes.keys()) != set(AXIS_NAMES):
        errors.append(f"axes must be exactly {sorted(AXIS_NAMES)}")

    for name, axis in axes.items():
        if "level" not in axis or "confidence" not in axis or "evidence" not in axis:
            errors.append(f"axis {name!r} missing level/confidence/evidence")
            continue
        if axis["level"] not in AXIS_LEVELS[name]:
            errors.append(
                f"axis {name!r} level {axis['level']!r} not in {sorted(AXIS_LEVELS[name])}"
            )
        if not isinstance(axis["confidence"], (int, float)) or not 0.0 <= axis["confidence"] <= 1.0:
            errors.append(f"axis {name!r} confidence out of [0,1]: {axis['confidence']}")

    # Ethological invariant: semantics may be validated only by Track C biology.
    sv_level = axes["semantic_validation"]["level"]
    if report["validated_semantics"] and sv_level != "playback-supported":
        errors.append(
            "validated_semantics=True requires semantic_validation level "
            "'playback-supported' (Track C biology)"
        )
    if sv_level != "playback-supported" and report["validated_semantics"] is not False:
        errors.append("validated_semantics must be False without Track C playback evidence")

    for entry in report["falsification"]:
        for key in ("axis", "level", "makes_positive_claim", "falsification_tests"):
            if key not in entry:
                errors.append(f"falsification entry missing {key!r}: {entry}")
        for test in entry.get("falsification_tests", []):
            if "label" not in test or "track" not in test:
                errors.append(f"falsification test missing label/track: {test}")
            elif test["track"] not in FALSIFICATION_TRACKS:
                errors.append(f"unknown falsification track {test['track']!r}")

    # Optional domain-gate block (diagnostics.domain_gate). Not required; if present,
    # light-validate its shape so a gate-rejected report stays self-describing.
    if "domain" in report:
        domain = report["domain"]
        if not isinstance(domain, dict):
            errors.append("'domain' must be an object")
        else:
            for key in ("dataset", "dataset_role", "verdict", "communication_status"):
                if key not in domain:
                    errors.append(f"domain block missing {key!r}")

    return errors


def is_valid(report: dict) -> bool:
    return not validate_report(report)
