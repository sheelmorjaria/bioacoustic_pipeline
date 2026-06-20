"""ConfigResolver — the single source of truth for a pipeline run.

Freezes, per run: the species NBD profile, the dataset role, the segment-representation
source (firewall-checked + compatibility-gated — never silent CPC), and the enabled
diagnostics. Reads an optional YAML config with Python fallback so it works without PyYAML.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from bioacoustic_pipeline.diagnostics.domain_gate import DATASET_ROLES, DEFAULT_ROLE
from bioacoustic_pipeline.embeddings.segment_representation import resolve_source
from bioacoustic_pipeline.ingestion.manifest import load_species_profiles
from bioacoustic_pipeline.pipeline.contracts import ResolvedConfig

# Phase-1 default diagnostic set (registry-registered). Phase 2 adds
# controlled_alias / context_binding / acoustic_coherence by extending this list.
DEFAULT_DIAGNOSTICS = ("sequence_sufficiency", "lempel_ziv", "motif")


class ConfigResolver:
    """Resolve a frozen :class:`ResolvedConfig` for a run."""

    @staticmethod
    def resolve(
        *,
        species: str,
        dataset: str | None = None,
        config: dict | None = None,
        segment_source=None,
        seed: int = 0,
        diagnostics=None,
    ) -> ResolvedConfig:
        config = config or {}
        profiles = load_species_profiles()
        if species not in profiles:
            raise ValueError(f"unknown species {species!r}; known: {sorted(profiles)}")
        profile = profiles[species]

        role = DATASET_ROLES.get(dataset, DEFAULT_ROLE) if dataset else DEFAULT_ROLE

        # Segment representation: explicit arg > config default > None (caller-supplied
        # tokenizer). When resolved, it is firewall-checked for the species.
        src = segment_source
        if src is None and config.get("segment_source"):
            src = resolve_source(species, config["segment_source"])

        diags = (
            list(diagnostics)
            if diagnostics
            else list(config.get("diagnostics") or DEFAULT_DIAGNOSTICS)
        )

        return ResolvedConfig(
            species=species,
            dataset=dataset,
            dataset_role=role,
            profile=profile,
            segment_source=src,
            enabled_diagnostics=diags,
            seed=seed,
        )
