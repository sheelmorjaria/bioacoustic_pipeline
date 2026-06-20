"""Typed contracts for the canonical pipeline — the spine everything plugs into.

Phase 1 of the integration refactor: replace ad-hoc parallel lists and script-specific
output shapes with a small set of typed contracts. ``run_pipeline`` (legacy) becomes a
thin wrapper over ``pipeline.run``, which returns a :class:`PipelineResult`; the
:class:`DiagnosticRegistry` emits :class:`DiagnosticResult` objects; the
:class:`ArtifactWriter` consumes a :class:`PipelineResult` to emit one canonical bundle.

Pure stdlib; no model deps. Pure data — no behavior beyond serialization helpers.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Diagnostic status vocabulary. ``computed`` = ran and produced a value;
# ``not_applicable`` = domain gate refused this axis (out-of-domain corpus);
# ``not_evaluable`` = sequence-sufficiency gate abstained (too sparse to test);
# ``skipped`` = disabled in config / prerequisite missing.
STATUS_COMPUTED = "computed"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_NOT_EVALUABLE = "not_evaluable"
STATUS_SKIPPED = "skipped"
DIAGNOSTIC_STATUSES = (STATUS_COMPUTED, STATUS_NOT_APPLICABLE, STATUS_NOT_EVALUABLE, STATUS_SKIPPED)


@dataclass
class DiagnosticResult:
    """One diagnostic's outcome in a pipeline run."""

    name: str
    status: str  # one of DIAGNOSTIC_STATUSES
    value: dict | None = None  # the underlying result's as_dict() payload (None when abstaining)
    axis: str | None = None  # verdict axis this feeds (if any)
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in DIAGNOSTIC_STATUSES:
            raise ValueError(f"unknown diagnostic status {self.status!r}")

    @property
    def computed(self) -> bool:
        return self.status == STATUS_COMPUTED

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "axis": self.axis,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass
class ResolvedConfig:
    """What ``ConfigResolver`` froze for a run — the single source of truth."""

    species: str
    dataset: str | None
    dataset_role: str
    profile: dict
    segment_source: object | None  # embeddings.segment_representation.SegmentSource | None
    enabled_diagnostics: list[str] = field(default_factory=list)
    seed: int = 0

    def as_dict(self) -> dict:
        ss = self.segment_source
        return {
            "species": self.species,
            "dataset": self.dataset,
            "dataset_role": self.dataset_role,
            "profile": dict(self.profile),
            "segment_source": (ss.as_dict() if hasattr(ss, "as_dict") else ss),
            "enabled_diagnostics": list(self.enabled_diagnostics),
            "seed": self.seed,
        }


@dataclass
class CorpusManifest:
    """Light corpus description (dataset id, role, audio paths, annotations)."""

    dataset: str | None = None
    dataset_role: str | None = None
    audio_paths: list = field(default_factory=list)
    annotations: str | None = None

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "dataset_role": self.dataset_role,
            "n_files": len(self.audio_paths),
            "annotations": self.annotations,
        }


@dataclass
class PipelineResult:
    """The canonical typed return of ``pipeline.run``."""

    report: dict  # schema-valid verdict
    trace: dict
    manifest: dict  # reproducibility manifest (git/seed/versions/hashes)
    domain_gate: dict  # DomainGateResult.as_dict() (applicability mask)
    segments: list[dict]  # JSON-safe per-segment records
    dual_stream: list[dict]  # DualStreamSegment.as_event() list (the typed event spine)
    sequences: list = field(default_factory=list)  # per-file token sequences (writer/diagnostics)
    diagnostics: dict = field(default_factory=dict)  # name -> DiagnosticResult
    config: dict | None = None  # ResolvedConfig.as_dict()
    artifacts: dict | None = None  # {name: path} filled by ArtifactWriter
    playback_manifest: dict | None = None

    def as_tuple(self) -> tuple[dict, dict]:
        """Backward-compatible ``(report, trace)`` view for the legacy run_pipeline wrapper."""
        return self.report, self.trace

    def diagnostics_as_dict(self) -> dict:
        return {name: dr.as_dict() for name, dr in self.diagnostics.items()}
