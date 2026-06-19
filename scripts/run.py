#!/usr/bin/env python3
"""High-level orchestration: token stream + boundaries -> schema-valid verdict.

Stitches the upstream stages (segmentation boundary types, diagnostics LZ /
alias / null models) into the verdict engine, emitting the canonical
Evidence-Profile + Falsification report (validated against the schema).

This is the integration seam: every upstream module feeds here, and the output
must satisfy ``verdict_engine.schema``. Run on real audio once checkpoints are
in place (see scripts/validate_track_a.py).

Usage:
    python scripts/run.py --species bat --tokens-file tokens.json

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity  # noqa: E402
from bioacoustic_pipeline.verdict_engine import build_report  # noqa: E402


def synthesize_signals(tokens, boundary_types, sequences=None, with_null=False):
    """Reduce raw outputs to the primitive signals the verdict engine consumes."""
    lz = lempel_ziv_complexity(tokens[:5000]).normalized
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
                r = detect_functional_aliases(seqs, similarity_threshold=0.80, min_instances=10)
                e = len(r.singleton_tokens) + r.n_merged
                return r.n_merged / max(e, 1)

            res = permutation_test(sequences, _density, within_file_shuffle, n_permutations=100)
            alias_pvalue = res.p_value

    return {
        "lz_normalized": lz,
        "n_hard": n_hard,
        "n_soft": n_soft,
        "n_total_boundaries": n_hard + n_soft,
        "alias_density": alias_density,
        "alias_pvalue": alias_pvalue,
    }


def run_report(species, tokens, boundary_types, sequences=None, with_null=False) -> dict:
    signals = synthesize_signals(tokens, boundary_types, sequences, with_null)
    return build_report(species, **signals)


def main():
    ap = argparse.ArgumentParser(description="Token stream -> verdict report")
    ap.add_argument("--species", default="bat")
    ap.add_argument("--tokens-file", required=True,
                    help="JSON: {tokens:[...], boundary_types:[...], sequences:[[...]]}")
    ap.add_argument("--with-null", action="store_true", help="run permutation null on aliases")
    args = ap.parse_args()

    data = json.loads(Path(args.tokens_file).read_text())
    report = run_report(
        args.species,
        data["tokens"],
        data.get("boundary_types", []),
        sequences=data.get("sequences"),
        with_null=args.with_null,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
