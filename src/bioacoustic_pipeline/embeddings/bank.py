"""Embedding bank — multi-view, firewall-gated, cache-backed orchestration.

Holds multiple backbones; emits one embedding per backbone per segment, each
carrying provenance. Only backbones that loaded AND pass the firewall
(Use.EMBEDDING + cross-species) are active; the rest are reported as skipped.
Every embedding is content-addressed through the cache so it is never recomputed.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioacoustic_pipeline.embeddings.cache import EmbeddingCache
from bioacoustic_pipeline.embeddings.firewall import Firewall, Use, default_firewall


@dataclass
class BackboneView:
    name: str
    vector: np.ndarray
    provenance: dict


class EmbeddingBank:
    def __init__(
        self,
        backbones,
        *,
        firewall: Firewall | None = None,
        target_species: str | None = None,
        cache: EmbeddingCache | str | None = None,
    ):
        self.firewall = firewall or default_firewall()
        self.target_species = target_species
        self.cache = (
            cache if isinstance(cache, EmbeddingCache) or cache is None
            else EmbeddingCache(cache)
        )
        self._all = list(backbones)
        self.backbones = []
        self._skipped: list[tuple[str, str]] = []
        for b in self._all:
            if not b.available:
                self._skipped.append((b.name, b.load_error or "unavailable"))
                continue
            try:
                self.firewall.assert_allowed(b.provenance_name, Use.EMBEDDING)
                if target_species is not None:
                    self.firewall.assert_cross_species(b.provenance_name, target_species)
            except Exception as exc:  # noqa: BLE001
                self._skipped.append((b.name, f"firewall: {exc}"))
                continue
            self.backbones.append(b)

    @property
    def active_names(self) -> list[str]:
        return [b.name for b in self.backbones]

    @property
    def skipped(self) -> list[tuple[str, str]]:
        return list(self._skipped)

    def _provenance(self, backbone) -> dict:
        p = self.firewall.get(backbone.provenance_name)
        return {
            "provenance_name": p.model_name,
            "embed_dim": backbone.embed_dim,
            "pretraining_domain": p.pretraining_domain,
            "contains_human_speech": p.contains_human_speech,
            "contains_text_supervision": p.contains_text_supervision,
        }

    def embed_all(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for b in self.backbones:
            if self.cache is not None:
                out[b.name] = self.cache.embed(b.name, audio, sr, b.embed)
            else:
                out[b.name] = b.embed(audio, sr)
        return out

    def views(self, audio: np.ndarray, sr: int) -> list[BackboneView]:
        return [
            BackboneView(name=b.name, vector=vec, provenance=self._provenance(b))
            for b, vec in zip(self.backbones, self.embed_all(audio, sr).values())
        ]

    def report(self) -> dict:
        return {
            "target_species": self.target_species,
            "active": [{"name": b.name, **self._provenance(b)} for b in self.backbones],
            "skipped": [{"name": n, "reason": r} for n, r in self._skipped],
        }
