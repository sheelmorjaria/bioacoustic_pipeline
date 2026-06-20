"""Motif Miner — discovers recurring token, alias, acoustic, and turn-taking patterns.

Motifs are statistically tested sequence hypotheses, NOT syntax. They must survive
null models, bootstrap stability, and behavioral validation before any structural claim.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from .scoring import Motif, MotifReport, npmi, entropy_reduction, score_motif
from .token_motifs import mine_ngrams, build_motifs
from .alias_motifs import collapse_to_alias, mine_alias_motifs
from .acoustic_motifs import cluster_fingerprints, mine_acoustic_motifs
from .turn_motifs import build_turn_events, mine_turn_motifs
from .affective_motifs import mine_affective_motifs
from .nulls import motif_support_null
from .report import format_motif_report

__all__ = [
    "Motif", "MotifReport", "npmi", "entropy_reduction", "score_motif",
    "mine_ngrams", "build_motifs",
    "collapse_to_alias", "mine_alias_motifs",
    "cluster_fingerprints", "mine_acoustic_motifs",
    "build_turn_events", "mine_turn_motifs",
    "mine_affective_motifs",
    "motif_support_null", "format_motif_report",
]
