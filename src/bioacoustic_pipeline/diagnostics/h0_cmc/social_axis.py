"""Social axis — separate social covariates from generic context (H0-CMC Step 7).

The H0-CMC framework separates SOCIAL context (who is vocalising, to whom, their
relationship) from GENERIC context (behavioural state, environment, time of day). The bat
§9 result shows social covariates carry most of the multimodal synergy (D=4.3× ≈ G=4.7×;
A alone = 2.7×). This module provides the data containers + a split function that takes a
raw label dict and separates the two axes.

Lightweight dataclasses (no sklearn dependency). NOT validated meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SocialContext:
    """Social-axis covariates (who / to whom / relationship)."""

    emitter_id: str | None = None
    addressee_id: str | None = None
    relationship: str | None = None  # kin, mate, rival, stranger
    familiarity: float | None = None  # 0–1, prior interaction frequency
    audience_composition: list[str] | None = None  # who else is present
    spatial_proximity: float | None = None  # metres to nearest conspecific
    group_size: int | None = None

    def to_onehot(self, vocab: dict | None = None) -> np.ndarray:
        """Flatten to a feature vector (emitter/addressee one-hot + scalars)."""
        parts = []
        for attr in ("emitter_id", "addressee_id", "relationship"):
            v = getattr(self, attr)
            if v is not None and vocab and attr in vocab:
                oh = np.zeros(len(vocab[attr]), dtype=np.float32)
                if v in vocab[attr]:
                    oh[vocab[attr][v]] = 1.0
                parts.append(oh)
        for attr in ("familiarity", "spatial_proximity", "group_size"):
            v = getattr(self, attr)
            parts.append(np.array([float(v) if v is not None else 0.0], dtype=np.float32))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


@dataclass
class GenericContext:
    """Non-social context (behavioural / environmental / individual state)."""

    behavioral_state: str | None = None  # feeding, resting, moving, aggression
    environmental: str | None = None  # indoor, outdoor, novel enclosure
    individual_state: str | None = None  # hungry, stressed, reproductive
    time_of_day: float | None = None  # 0–24h
    recording_location: str | None = None

    def to_onehot(self, vocab: dict | None = None) -> np.ndarray:
        parts = []
        for attr in ("behavioral_state", "environmental", "individual_state", "recording_location"):
            v = getattr(self, attr)
            if v is not None and vocab and attr in vocab:
                oh = np.zeros(len(vocab[attr]), dtype=np.float32)
                if v in vocab[attr]:
                    oh[vocab[attr][v]] = 1.0
                parts.append(oh)
        parts.append(
            np.array([float(self.time_of_day) if self.time_of_day else 0.0], dtype=np.float32)
        )
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def split_context(labels: dict) -> tuple[SocialContext, GenericContext]:
    """Split a raw label dict into social + generic axes.

    Keys recognised as SOCIAL: emitter, addressee, relationship, familiarity, audience,
    proximity, group_size. Everything else maps to generic.
    """
    _SOCIAL_KEYS = {
        "emitter": "emitter_id",
        "addressee": "addressee_id",
        "relationship": "relationship",
        "familiarity": "familiarity",
        "audience": "audience_composition",
        "proximity": "spatial_proximity",
        "group_size": "group_size",
    }
    social = SocialContext()
    generic = GenericContext()
    for k, v in labels.items():
        if k in _SOCIAL_KEYS:
            setattr(social, _SOCIAL_KEYS[k], v)
        elif k == "behavior":
            generic.behavioral_state = v
        elif k in ("environment", "context", "location"):
            generic.environmental = v
        elif k == "time":
            generic.time_of_day = v
    return social, generic
