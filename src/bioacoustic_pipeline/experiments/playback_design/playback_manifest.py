"""Playback manifest assembler — the key output of Phase 1.

Assembles the dataset-level manifest: experiment, species, stimuli (with token sequences,
affect-trajectory descriptors, spatial condition, audio file, naturalness preflight), the
predicted behavior (a falsifiable hypothesis, NOT a claim), the response measures the
live side records, and the explicit caveats that keep the pipeline honest.

Writes canonical JSON via ``hashing.write_canonical_json``.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

EXPERIMENTS = (
    "motif_disruption",
    "alias_substitution",
    "affect_dissociation",
    "spatial_mismatch",
    "context_mismatch",
)

# Default falsifiable predictions per experiment type. These are HYPOTHESES for live
# (Track C) testing — the pipeline never asserts the animal will respond this way.
_DEFAULT_PREDICTIONS = {
    "motif_disruption": {
        "natural_vs_disrupted": "if the motif is behaviorally meaningful, natural motifs "
        "elicit a different response profile than reversed/shuffled/"
        "deleted motifs",
    },
    "alias_substitution": {
        "natural_vs_alias": "if B and B' are true functional aliases, A B C and A B' C "
        "elicit similar responses; the non-alias control A X C differs",
    },
    "affect_dissociation": {
        "natural_vs_flattened": "if the affect trajectory carries graded state, response "
        "intensity tracks the contour even with the token sequence held fixed",
    },
    "spatial_mismatch": {
        "correct_vs_wrong_direction": "if spatial-social structure matters, correct-direction "
        "unicast elicits stronger addressee-specific response",
    },
    "context_mismatch": {
        "matched_vs_mismatched": "if the motif is context-bound, behavioral response depends "
        "on whether playback context matches the learned context",
    },
}

# Explicit avoid-list caveats — keep the computational side from overclaiming.
CAVEATS = (
    "No 'bioacoustic Turing test' claim: acoustic naturalness screening is not animal acceptance.",
    "Motifs are statistically-tested sequence hypotheses, not 'grammar' or 'phrases'.",
    "No semantic decoding: meaning requires Track C (playback / receiver-response) biology.",
    "Affect is a graded acoustic-state CANDIDATE, not validated affect.",
    "Phase 1 stimuli are exemplar concatenations (real segment audio + ISI + crossfade), "
    "not vocoder/DDSP output; affect-morphed audio is not yet rendered.",
)


def build_manifest(
    experiment: str,
    species: str,
    stimuli: list[dict],
    *,
    predictions: dict | None = None,
    extra_caveats: tuple[str, ...] = (),
    source_motifs: list[dict] | None = None,
) -> dict:
    """Assemble the playback manifest.

    Args:
        experiment: one of ``EXPERIMENTS``.
        species: nominal species label.
        stimuli: list of per-stimulus dicts (stimulus_id, condition, token_sequence,
            affect_trajectory, spatial_condition, audio_file, naturalness_preflight, ...).
        predictions: override the default falsifiable predictions for this experiment.
        extra_caveats: additional caveats appended to the standard set.
        source_motifs: the motifs the stimuli were built from (provenance / audit).
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}; choose from {EXPERIMENTS}")
    from bioacoustic_pipeline.experiments.playback_design.response_schema import RESPONSE_MEASURES  # noqa: PLC0415

    return {
        "experiment": experiment,
        "species": species,
        "phase": "phase1_exemplar_concatenation",
        "n_stimuli": len(stimuli),
        "source_motifs": source_motifs or [],
        "stimuli": stimuli,
        "predicted_behavior": predictions
        if predictions is not None
        else _DEFAULT_PREDICTIONS[experiment],
        "response_measures": list(RESPONSE_MEASURES),
        "caveats": list(CAVEATS) + list(extra_caveats),
    }
