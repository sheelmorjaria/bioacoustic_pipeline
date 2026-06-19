"""Pipeline orchestration: upstream outputs -> schema-valid verdict report.

The integration seam. Reduces segmentation/diagnostics outputs to the primitive
signals the verdict engine consumes, then assembles + self-validates the
canonical Evidence-Profile + Falsification report.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity
from bioacoustic_pipeline.verdict_engine import build_report


def synthesize_signals(tokens, boundary_types, sequences=None, with_null=False) -> dict:
    """Reduce raw stage outputs to the verdict engine's primitive signals."""
    lz = lempel_ziv_complexity(list(tokens)[:5000]).normalized
    n_hard = sum(1 for b in boundary_types if b == "hard")
    n_soft = sum(1 for b in boundary_types if b == "soft")

    alias_density, alias_pvalue = 0.0, None
    if sequences and len(sequences) > 5:
        from bioacoustic_pipeline.diagnostics.alias_detector import (
            detect_functional_aliases,
        )

        rep = detect_functional_aliases(sequences, similarity_threshold=0.80, min_instances=10)
        eligible = len(rep.singleton_tokens) + rep.n_merged
        alias_density = rep.n_merged / max(eligible, 1)
        if with_null:
            from bioacoustic_pipeline.diagnostics.null_models import (
                permutation_test,
                within_file_shuffle,
            )

            def _density(seqs):
                r = detect_functional_aliases(
                    seqs, similarity_threshold=0.80, min_instances=10
                )
                e = len(r.singleton_tokens) + r.n_merged
                return r.n_merged / max(e, 1)

            alias_pvalue = permutation_test(
                sequences, _density, within_file_shuffle, n_permutations=100
            ).p_value

    return {
        "lz_normalized": lz,
        "n_hard": n_hard,
        "n_soft": n_soft,
        "n_total_boundaries": n_hard + n_soft,
        "alias_density": alias_density,
        "alias_pvalue": alias_pvalue,
    }


def run_report(species, tokens, boundary_types, sequences=None, with_null=False) -> dict:
    """Assemble + self-validate the canonical verdict report."""
    return build_report(species, **synthesize_signals(
        tokens, boundary_types, sequences, with_null
    ))
