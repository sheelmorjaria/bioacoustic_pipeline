"""Playback-design (Phase 1) — turn discovered structures into falsifiable playback stimuli.

Phase 1 is **exemplar-based** (no neural synthesis): it concatenates real per-token segment
audio for a motif's token pattern, generates controlled sequence variants (reverse/shuffle/
delete/insert/random/alias-substitute), preflights them for acoustic naturalness, and emits a
playback manifest. Every output is a **falsifiable hypothesis** for live (Track C) validation
— never a Turing-test/grammar/semantic-decoding claim. Affect is a manifest-level
descriptor; rendering affect-shifted audio is Phase 2/3 (needs a trained vocoder).

Modules:
  candidate_selector  — pick high-confidence motifs + per-token natural exemplars
  motif_variants      — token-sequence perturbations (reverse/shuffle/delete/insert/random)
  stimulus_builder    — exemplar concatenation + ISI + envelope crossfade -> .wav
  alias_substitution  — swap a token for a functional-alias group member
  affect_morphing     — natural + flattened/exaggerated/reversed affect TRAJECTORY descriptors
  spatial_router      — VBAP gains + spatial_condition labels (unicast/broadcast, direction)
  naturalness_preflight — fingerprint + spectrogram + DTW distance vs natural exemplars
  playback_manifest   — assemble the manifest JSON (stimuli + predictions + caveats)
  response_schema     — declared behavioral-response schema for the live side

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from bioacoustic_pipeline.experiments.playback_design.affect_morphing import (
    NOT_AVAILABLE as AFFECT_NOT_AVAILABLE,
    affect_trajectory,
    morph_descriptors,
)
from bioacoustic_pipeline.experiments.playback_design.alias_substitution import (
    alias_substitute_pattern,
    nonalias_control_pattern,
)
from bioacoustic_pipeline.experiments.playback_design.candidate_selector import (
    TokenExemplar,
    select_candidates,
)
from bioacoustic_pipeline.experiments.playback_design.motif_variants import (
    random_control,
    reverse,
    shuffle,
    token_delete,
    token_insert,
)
from bioacoustic_pipeline.experiments.playback_design.naturalness_preflight import (
    preflight_stimulus,
)
from bioacoustic_pipeline.experiments.playback_design.playback_manifest import (
    build_manifest,
)
from bioacoustic_pipeline.experiments.playback_design.response_schema import (
    RESPONSE_MEASURES,
    empty_response_record,
)
from bioacoustic_pipeline.experiments.playback_design.spatial_router import (
    SPATIAL_CONDITIONS,
    spatial_condition,
    vbap_gains,
)
from bioacoustic_pipeline.experiments.playback_design.stimulus_builder import (
    build_stimulus_audio,
)

__all__ = [
    "TokenExemplar",
    "select_candidates",
    "reverse",
    "shuffle",
    "token_delete",
    "token_insert",
    "random_control",
    "alias_substitute_pattern",
    "nonalias_control_pattern",
    "build_stimulus_audio",
    "affect_trajectory",
    "morph_descriptors",
    "AFFECT_NOT_AVAILABLE",
    "vbap_gains",
    "spatial_condition",
    "SPATIAL_CONDITIONS",
    "preflight_stimulus",
    "build_manifest",
    "RESPONSE_MEASURES",
    "empty_response_record",
]
