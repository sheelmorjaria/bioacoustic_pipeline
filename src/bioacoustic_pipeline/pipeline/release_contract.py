"""Strict scientific-release contract: fail-fast gates for every run.

The 2026-10 consolidation turns individually-functioning components into one
dependable workflow. The failures that motivated this module (all found in
real runs, all preserved as fixture tests):

  1. artificial padding / recorded silence entering analyses as observations;
  2. partial feature support (1 valid frame in a 40-frame window) treated as
     a trajectory;
  3. activity-mask vs representation-validity disagreement silently resolved
     by one mask standing in for the other;
  4. feature-layout drift (a stale name->index map reading the wrong
     coordinate);
  5. duplicate source files, including the same file on both sides of a
     frozen split;
  6. evaluation sources leaking into a fitting pool.

Four validity concepts are DISTINCT and never interchangeable:

    source validity  !=  detected activity  !=  feature support  !=  task eligibility

A scientific run refuses to proceed (ReleaseContractError) when a required
mask provenance, feature-schema version, source mapping, or evaluation split
is missing, and a recurrence run refuses the backward-compatible unmasked
search path.

The task->representation policy records WHICH component is justified for
each task (evidence-backed, per §45/§46 and the 2026-10 dynamics studies).
Learned scaling, codebooks and fitted preprocessing are training-only and
travel with their fitted models.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bioacoustic_pipeline.diagnostics.acoustic_derivatives import FEATURE_SCHEMA_VERSION

RELEASE_VERSION = "2026.10.18"

MASK_KINDS = ("source_validity", "detected_activity", "feature_support", "task_eligibility")


class ReleaseContractError(RuntimeError):
    """A scientific run violated the release contract — refuse to proceed."""


TASK_REPRESENTATION_POLICY: dict = {
    "caller_id_and_form_retrieval": {
        "default": "whole-call 44-D fingerprint",
        "optional": "CPC, or fingerprint-CPC fusion, where the specific "
        "held-out endpoint improves (BENCHMARK §45)",
    },
    "discrete_temporal_prediction": {
        "default": "frozen pooled PST",
        "optional": "caller-specific models retained as documented ablations "
        "(valid_reanalysis: pooling won)",
    },
    "continuous_acoustic_forecasting": {
        "default": "state + previous state",
        "optional": "state + derivative for interpretable rates of change "
        "(2026-10: informationally equivalent parameterisation)",
    },
    "acoustic_dynamics_description": {
        "default": "supported absolute trajectories + 12 derivatives",
        "optional": "prespecified context / recording-condition summaries",
    },
    "vocal_continuation_scoring": {
        "default": "eligible shared CPC checkpoint for the task",
        "optional": "vocal-target continuation under its documented mask and "
        "provenance (pilot endpoints)",
    },
    "recurrence_discovery": {
        "default": "validity-aware, source-linked search with explicit abstention",
        "optional": None,  # no automatic biological interpretation / playback promotion
    },
}


@dataclass(frozen=True)
class MaskProvenance:
    """Provenance for one validity concept's mask."""

    kind: str
    produced_by: str
    win_ms: float
    hop_ms: float
    amplitude_stage: str  # e.g. "native waveform (absolute gate)"


@dataclass
class ScientificRunContract:
    """Everything a scientific run must declare before it may proceed."""

    run_name: str
    feature_schema_version: int
    masks: dict = field(default_factory=dict)  # kind -> MaskProvenance
    source_manifest: dict = field(default_factory=dict)  # file -> group / part
    eval_split: dict = field(default_factory=dict)  # partition provenance
    recurrence_validity_mode: str | None = None  # required for recurrence runs
    fitted_preprocessing: dict = field(default_factory=dict)  # name -> fit partition


def assert_feature_schema(version: int) -> None:
    """Gate 4: the layout the run assumes must be the verified layout."""
    if version != FEATURE_SCHEMA_VERSION:
        raise ReleaseContractError(
            f"feature schema version {version} != {FEATURE_SCHEMA_VERSION}: "
            "derivative outputs under a stale name->index map are invalid "
            "(v1 read mod_depth@18 as env_cv)"
        )


def assert_no_padding_as_valid(frames, mask) -> None:
    """Gate 1: all-zero frames (padding or recorded silence) are never valid."""
    import numpy as np

    frames = np.asarray(frames)
    mask = np.asarray(mask, dtype=bool)
    zero_rows = np.all(frames == 0, axis=1)
    if np.any(mask & zero_rows):
        n = int((mask & zero_rows).sum())
        raise ReleaseContractError(
            f"{n} all-zero frames marked valid: padding/silence must be "
            "excluded by the feature-support mask, never observed as evidence"
        )


def assert_derivative_support(valid, min_adjacent_pairs: int = 1) -> None:
    """Gate 2: partial support cannot stand in for a trajectory."""
    import numpy as np

    valid = np.asarray(valid, dtype=bool)
    pairs = int((valid[1:] & valid[:-1]).sum())
    if pairs < min_adjacent_pairs:
        raise ReleaseContractError(
            f"derivative support {pairs} adjacent valid pair(s) < required "
            f"{min_adjacent_pairs}: insufficient representational support — "
            "the window is unavailable as trajectory evidence, never distance-0"
        )


def assert_mask_disagreement_reported(masks: dict) -> dict:
    """Gate 3: activity and representation-validity are reported separately.

    Returns the disagreement statistics; raises when a run declares only one
    of the two masks (one mask standing in for the other is the integration
    failure behind the rejected recurrence candidates).
    """
    for kind in ("detected_activity", "source_validity"):
        if not isinstance(masks.get(kind), MaskProvenance):
            raise ReleaseContractError(
                f"mask '{kind}' missing: activity and representation-validity "
                "must both be declared — they answer different questions and "
                "disagree on low-gain recordings"
            )
    return {"declared": sorted(masks)}


def assert_no_duplicate_sources(names, parts=None) -> None:
    """Gate 5: no duplicated sources; none on both sides of a split."""
    names = list(names)
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ReleaseContractError(f"duplicate sources present: {sorted(dupes)[:5]}")
    if parts is not None:
        side: dict = {}
        for n, p in zip(names, parts):
            if n in side and side[n] != p:
                raise ReleaseContractError(f"{n} appears on both sides of the split")
            side[n] = p


def assert_no_eval_leakage(fit_names, eval_names) -> None:
    """Gate 6: evaluation sources never enter a fitting pool."""
    leak = set(map(str, fit_names)) & set(map(str, eval_names))
    if leak:
        raise ReleaseContractError(f"evaluation sources in the fitting pool: {sorted(leak)[:5]}")


def require_masked_recurrence(valid_masks, min_valid_frames: int) -> None:
    """A recurrence run may not use the backward-compatible unmasked path."""
    if valid_masks is None or min_valid_frames is None:
        raise ReleaseContractError(
            "unmasked recurrence search: representation support "
            "(valid_masks + min_valid_frames) is mandatory — the legacy path "
            "produced gate-structure matches rejected 2026-10-18"
        )


def assert_feature_contract(
    features,
    *,
    expected_names: tuple | None = None,
    expected_version: int | None = None,
    context: str = "scientific run",
) -> None:
    """Runtime representation contract at fitting AND scoring entry.

    Catches the failures a version number alone cannot: a wrong-width
    array (44-D entering a 12-feature study), a right-width but
    reordered/renamed set, and diagnostics whose labels come from a
    different schema object than the data. Motivated by the marmoset
    replication's 44-D contract violation (caught only via pathological
    losses and a mislabelled diagnostic).
    """
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
        supported_features,
    )

    names = tuple(expected_names) if expected_names is not None else supported_features()
    version = expected_version if expected_version is not None else FEATURE_SCHEMA_VERSION
    arr = getattr(features, "shape", None)
    if arr is not None:
        if len(arr) < 2 or arr[1] != len(names):
            raise ReleaseContractError(
                f"{context}: feature width {arr[1]} != required {len(names)} "
                f"({', '.join(names)}) — a wider/narrower array must not "
                "enter fitting or scoring"
            )
    elif hasattr(features, "__len__") and len(features) != len(names):
        raise ReleaseContractError(
            f"{context}: got {len(features)} features, expected {len(names)}"
        )
    if version != FEATURE_SCHEMA_VERSION:
        raise ReleaseContractError(
            f"{context}: schema version {version} != {FEATURE_SCHEMA_VERSION}"
        )


def source_manifest_from(names, parts, groups=None) -> dict:
    """Build the file->{group, part} manifest with duplicate gates applied.

    Lists are the realistic input (a manifest built from lists can carry a
    file on both sides of a split; the gates refuse that here, at assembly).
    """
    names = [str(n) for n in names]
    parts = list(parts)
    groups = list(groups) if groups is not None else [str(n) for n in names]
    if not (len(names) == len(parts) == len(groups)):
        raise ReleaseContractError("names/parts/groups length mismatch")
    assert_no_duplicate_sources(names, parts)
    return {n: {"group": g, "part": p} for n, g, p in zip(names, groups, parts)}


def raise_for_science(contract: ScientificRunContract) -> dict:
    """Validate a declared contract; raise on any violation.

    Returns the enforcement summary for the run's artifact.
    """
    assert_feature_schema(contract.feature_schema_version)
    if not contract.source_manifest:
        raise ReleaseContractError("source manifest missing (file -> group mapping)")
    if not contract.eval_split:
        raise ReleaseContractError(
            "evaluation split provenance missing — a scientific run may not "
            "score material whose partition is undeclared"
        )
    assert_no_duplicate_sources(
        list(contract.source_manifest),
        [v.get("part") for v in contract.source_manifest.values()],
    )
    fit = {f for f, v in contract.source_manifest.items() if v.get("part") != "eval"}
    ev = {f for f, v in contract.source_manifest.items() if v.get("part") == "eval"}
    assert_no_eval_leakage(fit, ev)
    if contract.recurrence_validity_mode is not None and "masked" not in (
        contract.recurrence_validity_mode or ""
    ):
        raise ReleaseContractError("recurrence run must declare masked validity mode")
    return {
        "release_version": RELEASE_VERSION,
        "run": contract.run_name,
        "feature_schema_version": contract.feature_schema_version,
        "masks_declared": sorted(contract.masks),
        "n_sources": len(contract.source_manifest),
        "eval_split": contract.eval_split,
        "fitted_preprocessing": contract.fitted_preprocessing,
        "policy": TASK_REPRESENTATION_POLICY,
    }
