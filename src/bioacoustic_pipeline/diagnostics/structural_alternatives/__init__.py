"""Structural alternatives to DUSTED — ablation-stage methods.

Four complementary views of vocal sub-structure that do NOT rely on DUSTED's token-sequence
alignment: RQA + Matrix Profile (continuous trajectory, tokenization-free) and PST + Transition
Network (discrete token sequences). These are COMPARATIVE-ABLATION candidates, not primary pipeline
tools — grouped here to signal they are candidates for regime-specific promotion (e.g. RQA for
graded repertoires) once a head-to-head vs DUSTED (``scripts/compare_structure_methods.py``) shows a
decisive win. Firewall-clean (all-species acoustic fingerprint / cluster IDs). NOT validated meaning.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from bioacoustic_pipeline.diagnostics.structural_alternatives.matrix_profile_motifs import (
    matrix_profile,
)
from bioacoustic_pipeline.diagnostics.structural_alternatives.probabilistic_suffix_tree import (
    fit_pst,
)
from bioacoustic_pipeline.diagnostics.structural_alternatives.rqa import rqa_for_trajectory
from bioacoustic_pipeline.diagnostics.structural_alternatives.transition_network import (
    transition_network_metrics,
)

__all__ = ["rqa_for_trajectory", "matrix_profile", "fit_pst", "transition_network_metrics"]
