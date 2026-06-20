"""Canonical typed pipeline — contracts + registry + orchestrator + artifact writer.

The integration spine: ``run(...)`` returns a :class:`PipelineResult` built from a
``DualStreamSegment`` event stream + a :class:`DiagnosticRegistry`. The legacy
``orchestration.run_pipeline`` is a thin backward-compat wrapper over ``run``.

Phase 1: contracts + spine + registry skeleton (sufficiency/LZ/motif) + applicability
mask + ArtifactWriter. Phase 2 adds more registered diagnostics (no orchestrator change).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from bioacoustic_pipeline.pipeline.artifacts import ARTIFACT_NAMES, ArtifactWriter
from bioacoustic_pipeline.pipeline.config import DEFAULT_DIAGNOSTICS, ConfigResolver
from bioacoustic_pipeline.pipeline.contracts import (
    CorpusManifest,
    DiagnosticResult,
    PipelineResult,
    ResolvedConfig,
)
from bioacoustic_pipeline.pipeline.orchestrator import run
from bioacoustic_pipeline.pipeline.registry import DiagnosticRegistry, default_registry

__all__ = [
    "run",
    "PipelineResult",
    "DiagnosticResult",
    "DiagnosticRegistry",
    "default_registry",
    "ConfigResolver",
    "ResolvedConfig",
    "CorpusManifest",
    "ArtifactWriter",
    "ARTIFACT_NAMES",
    "DEFAULT_DIAGNOSTICS",
]
