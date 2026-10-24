"""Ethological firewall — JSON-driven model provenance + use enforcement.

The single source of truth is ``configs/model_metadata.json`` (a registry of
every model's pretraining domain, human-speech / text-supervision flags, and
allowed/forbidden uses). This module parses it and HARD-FAILS on a forbidden
use (e.g. NatureLM for ``semantic_verdict``, a human model for an animal
species, any model for a meaning claim).

Unlike the prototype's hardcoded registry, this loads policy from JSON so the
firewall is data-driven and editable without code changes.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Use(str, Enum):
    EMBEDDING = "embedding"
    PROBE = "probe"
    TOKENIZE = "tokenize"
    AFFECT = "affect"
    RETRIEVAL = "retrieval"
    ANNOTATION_SUGGESTION = "annotation_suggestion"
    VERDICT_WEIGHT = "verdict_weight"
    SEMANTIC_VERDICT = "semantic_verdict"
    ANIMAL_MEANING_CLAIM = "animal_meaning_claim"


# Uses computation alone can NEVER authorize — only Track C biology can.
TRACK_C_ONLY = frozenset({Use.SEMANTIC_VERDICT, Use.ANIMAL_MEANING_CLAIM})

# Uses a caller-supervised model can NEVER have, independent of its
# allowed_use: caller labels inside training make the representation
# supervised metric learning, so any structure-discovery use is circular
# (the model was optimized to produce exactly that structure). A registry
# edit cannot re-authorize what the supervision semantics forbid.
CALLER_SUPERVISED_FORBIDDEN = frozenset({Use.TOKENIZE, Use.VERDICT_WEIGHT})


class ProvenanceViolation(Exception):
    """Raised when a requested use violates a model's provenance policy."""


@dataclass(frozen=True)
class ModelProvenance:
    model_name: str
    pretraining_domain: str
    contains_human_speech: bool
    contains_text_supervision: bool
    allowed_use: frozenset[Use]
    forbidden_use: frozenset[Use]
    applies_to_species: tuple[str, ...] = ()
    source: str = ""
    notes: str = ""
    contains_caller_supervision: bool = False

    def is_allowed(self, use: Use) -> bool:
        u = Use(use)
        if self.contains_caller_supervision and u in CALLER_SUPERVISED_FORBIDDEN:
            return False
        return u not in self.forbidden_use and u not in TRACK_C_ONLY and u in self.allowed_use


def _default_metadata_path() -> Path:
    env = os.environ.get("BIOACOUSTIC_MODEL_METADATA")
    if env:
        return Path(env)
    # configs/ lives at the repo root alongside src/.
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        cand = ancestor / "configs" / "model_metadata.json"
        if cand.exists():
            return cand
    return Path("configs/model_metadata.json")


class Firewall:
    """JSON-driven provenance registry + enforcement."""

    def __init__(self, metadata_path: str | Path | None = None):
        self.metadata_path = Path(metadata_path) if metadata_path else _default_metadata_path()
        self._registry: dict[str, ModelProvenance] = {}
        self._track_c_only: frozenset[str] = frozenset(
            u.value for u in TRACK_C_ONLY
        )
        self.reload()

    def reload(self) -> None:
        if not self.metadata_path.exists():
            raise ProvenanceViolation(
                f"firewall metadata not found at {self.metadata_path} — every model "
                f"must be registered before use"
            )
        raw = json.loads(self.metadata_path.read_text())
        self._track_c_only = frozenset(raw.get("track_c_only_uses", []))
        for name, m in raw.get("models", {}).items():
            self._registry[name] = ModelProvenance(
                model_name=name,
                pretraining_domain=m["pretraining_domain"],
                contains_human_speech=bool(m["contains_human_speech"]),
                contains_text_supervision=bool(m["contains_text_supervision"]),
                allowed_use=frozenset(Use(u) for u in m.get("allowed_use", [])),
                forbidden_use=frozenset(Use(u) for u in m.get("forbidden_use", [])),
                applies_to_species=tuple(m.get("applies_to_species", [])),
                source=m.get("source", ""),
                notes=m.get("notes", ""),
                contains_caller_supervision=bool(m.get("contains_caller_supervision", False)),
            )

    # ---- queries ----
    def get(self, model_name: str) -> ModelProvenance:
        if model_name not in self._registry:
            raise ProvenanceViolation(
                f"model '{model_name}' is not in the firewall registry "
                f"({self.metadata_path}) — register its provenance before use"
            )
        return self._registry[model_name]

    def is_allowed(self, model_name: str, use: Use) -> bool:
        try:
            return self.get(model_name).is_allowed(use)
        except ProvenanceViolation:
            return False

    def assert_allowed(self, model_name: str, use: Use) -> None:
        """Hard-fail unless model_name permits use."""
        prov = self.get(model_name)
        u = Use(use)
        if u.value in self._track_c_only:
            raise ProvenanceViolation(
                f"use '{u.value}' is Track-C-only (biology) — no model may do it; "
                f"'{model_name}' cannot."
            )
        if u in prov.forbidden_use:
            raise ProvenanceViolation(
                f"use '{u.value}' is FORBIDDEN for '{model_name}' "
                f"(domain={prov.pretraining_domain}, "
                f"text_supervision={prov.contains_text_supervision})"
            )
        if prov.contains_caller_supervision and u in CALLER_SUPERVISED_FORBIDDEN:
            raise ProvenanceViolation(
                f"use '{u.value}' is FORBIDDEN for caller-supervised "
                f"'{model_name}' — caller labels inside training make any "
                f"structure-discovery use circular (supervised metric "
                f"learning, not self-supervised representation)"
            )
        if u not in prov.allowed_use:
            raise ProvenanceViolation(
                f"use '{u.value}' not in the allowed set for '{model_name}': "
                f"{sorted(x.value for x in prov.allowed_use)}"
            )

    def assert_cross_species(self, model_name: str, target_species: str) -> None:
        prov = self.get(model_name)
        if prov.applies_to_species and target_species not in prov.applies_to_species:
            raise ProvenanceViolation(
                f"'{model_name}' (domain '{prov.pretraining_domain}', species "
                f"{list(prov.applies_to_species)}) must not be applied to "
                f"'{target_species}' — cross-species transfer forbidden"
            )

    def audit(self) -> list[dict]:
        return [
            {
                "model_name": p.model_name,
                "pretraining_domain": p.pretraining_domain,
                "contains_human_speech": p.contains_human_speech,
                "contains_text_supervision": p.contains_text_supervision,
                "allowed_use": sorted(u.value for u in p.allowed_use),
                "forbidden_use": sorted(u.value for u in p.forbidden_use),
                "applies_to_species": list(p.applies_to_species),
                "source": p.source,
                "notes": p.notes,
                "contains_caller_supervision": p.contains_caller_supervision,
            }
            for p in self._registry.values()
        ]


# Module-level singleton (lazy) for convenience.
_DEFAULT: Firewall | None = None


def default_firewall() -> Firewall:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Firewall()
    return _DEFAULT
