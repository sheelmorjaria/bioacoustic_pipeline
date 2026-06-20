"""Explicit segment-representation source + head compatibility check.

CPC is **one** segment representation, not a prerequisite. The real contract: a VQ-VAE /
β-VAE head must be assigned in the **same representation space it was trained on** (CPC /
BioMAE / acoustic-fingerprint / spectrogram / fused-bank). This module makes that dependency
explicit and checked:

  * ``SegmentSource`` — a chosen representation (category + provenance + dim + species).
  * ``resolve_source(species, category)`` — picks the firewall-legal provenance for a
    (category, species) pair (e.g. cpc+bat → bat_cpc; acoustic_fingerprint+* → all-species).
  * ``build_backbone(source)`` — constructs the backbone (lazy, guarded).
  * ``assert_compatible(head, source)`` — the gate: ``head.input_source == source.category``.

Each head (``AcousticTokenizer`` / ``SpeciesAcousticTokenizer`` /
``FingerprintTokenizerAdapter`` / ``BetaVAEBackbone``) declares an ``input_source`` category.
``BetaVAEBackbone.embed`` already returns ``mu`` (deterministic), so the assignment path stays
reproducible regardless of source.

``embedding_bank_fusion`` is a declared category whose fused backbone is NOT yet trained —
``build_backbone`` raises ``NotImplementedError`` rather than faking it.

Pure stdlib core; backbone factories are imported lazily (torch/onnx stay optional).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall

# Category = the representation space a head was trained in (not a provenance).
SOURCE_CATEGORIES = (
    "cpc",
    "biomae",
    "acoustic_fingerprint",
    "spectrogram",
    "embedding_bank_fusion",
)


@dataclass(frozen=True)
class SegmentSource:
    """A chosen segment representation for token/affect assignment."""

    category: str  # one of SOURCE_CATEGORIES
    provenance: str  # firewall registry key (e.g. "bat_cpc", "acoustic_fingerprint")
    species: str
    dim: int | None  # None = dynamic (fingerprint/spectrogram/fusion)
    view_class: str = "learned"

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "provenance": self.provenance,
            "species": self.species,
            "dim": self.dim,
            "view_class": self.view_class,
        }


# (category, species) -> (provenance, dim, view_class). All-species categories use "*".
_SPECIES_CPC = {"human": "human_cpc_phonetic", "bat": "bat_cpc", "marmoset": "marmoset_cpc"}


def _resolve_provenance(category: str, species: str) -> tuple[str, int | None, str]:
    if category == "cpc":
        prov = _SPECIES_CPC.get(species)
        if not prov:
            raise ValueError(f"cpc source has no registered species CPC for {species!r}")
        return prov, 256, "learned"
    if category == "biomae":
        return "biomae", 112, "learned"  # firewall asserts bat-only via assert_cross_species
    if category == "acoustic_fingerprint":
        return "acoustic_fingerprint", None, "interpretable"  # all-species, dim ~43 (dynamic)
    if category == "spectrogram":
        return "spectrogram", None, "interpretable"  # all-species, dim = n_mels
    if category == "embedding_bank_fusion":
        return "embedding_bank_fusion", None, "fused"
    raise ValueError(f"unknown source category {category!r}; choose from {SOURCE_CATEGORIES}")


def resolve_source(species: str, category: str, *, firewall=None) -> SegmentSource:
    """Resolve + firewall-check the representation for a (species, category) pair.

    Raises ``ProvenanceViolation`` if the resolved provenance is forbidden for the species
    (e.g. a species-CPC applied cross-species, or biomae off-bat).
    """
    fw = firewall or default_firewall()
    provenance, dim, view_class = _resolve_provenance(category, species)
    fw.assert_allowed(provenance, Use.EMBEDDING)
    fw.assert_cross_species(provenance, species)
    return SegmentSource(
        category=category,
        provenance=provenance,
        species=species,
        dim=dim,
        view_class=view_class,
    )


def build_backbone(source: SegmentSource):  # noqa: ANN201 (Backbone, torch-optional)
    """Construct the backbone for a resolved source (lazy import; torch/onnx stay optional).

    ``embedding_bank_fusion`` raises ``NotImplementedError`` — the fused backbone is not yet
    trained. Returns the backbone (call ``.available``/``.load_error`` to check it loaded).
    """
    prov = source.provenance
    if prov == "embedding_bank_fusion":
        raise NotImplementedError(
            "embedding_bank_fusion backbone is not yet trained/implemented; "
            "use a concrete category (cpc/biomae/acoustic_fingerprint/spectrogram)"
        )
    if prov == "human_cpc_phonetic":
        from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415

        return CPCBackbone()
    if prov in ("bat_cpc", "marmoset_cpc"):
        from bioacoustic_pipeline.embeddings.species_cpc_backbone import SpeciesCPCBackbone  # noqa: PLC0415

        return SpeciesCPCBackbone(name=prov, provenance_name=prov)
    if prov == "biomae":
        from bioacoustic_pipeline.embeddings.backbones import BioMAEBackbone  # noqa: PLC0415

        return BioMAEBackbone()
    if prov == "acoustic_fingerprint":
        from bioacoustic_pipeline.embeddings.acoustic_fingerprint_backbone import (  # noqa: PLC0415
            AcousticFingerprintBackbone,
        )

        return AcousticFingerprintBackbone()
    if prov == "spectrogram":
        from bioacoustic_pipeline.representation.spectrogram_backbone import SpectrogramBackbone  # noqa: PLC0415

        return SpectrogramBackbone()
    raise ValueError(f"no backbone factory for provenance {prov!r}")


def assert_compatible(head, source: SegmentSource, *, head_kind: str = "") -> None:
    """Gate: the head's declared ``input_source`` category must equal the source's category.

    Heads without an ``input_source`` attr (legacy/external) are skipped silently so the check
    is opt-in per head. Raises ``ValueError`` on a mismatch — e.g. feeding a fingerprint into a
    CPC-trained VQ-VAE.
    """
    hsrc = getattr(head, "input_source", None)
    if hsrc is None:
        return
    if hsrc != source.category:
        label = f"{head_kind} " if head_kind else ""
        raise ValueError(
            f"{label}head input_source {hsrc!r} != segment_representation source "
            f"{source.category!r} (head trained on a different representation space)"
        )


def assert_pairwise_compatible(vq_source: SegmentSource, affect_source: SegmentSource) -> None:
    """Optional: confirm the token head and the affect head share a representation space.

    They need not (a CPC-trained VQ-VAE + a fingerprint-derived affect are both legal), but the
    playback-design architecture pairs them; this helper makes an intentional pairing explicit.
    """
    if vq_source.category != affect_source.category:
        raise ValueError(
            f"vq_source {vq_source.category!r} and affect_source {affect_source.category!r} "
            f"differ — intentional cross-space pairing must be acknowledged by the caller"
        )


def load_segment_config(path) -> dict:
    """Load a segment_representation YAML config (lazy PyYAML; returns {} if PyYAML absent)."""
    from pathlib import Path  # noqa: PLC0415

    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        return {}
    with open(Path(path)) as fh:  # noqa: PTH123
        return yaml.safe_load(fh) or {}


# Re-export for callers that want to pass a custom resolver/factory.
BackboneFactory = Callable[[SegmentSource], object]
