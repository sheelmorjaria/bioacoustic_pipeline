#!/usr/bin/env python3
"""Cross-species repertoire taxonomy (Architecture v1.0 Step 3).

Classify each corpus's vocal repertoire into a TYPE that explains WHY it shows the structure
it does, from already-validated diagnostics. This is a SYNTHESIS table: the numbers are
curated from BENCHMARK §22 (cross-corpus DUSTED, uniform thr=0.15 HAC) + §13/§18/§23 (token↔
label Cramér's V) + §9 (bat social-lift), each cited in the profile's ``source``. The
classification logic lives in ``diagnostics.repertoire_taxonomy.classifier`` (pure stdlib,
unit-testable); this script assembles the validated per-corpus profiles and emits the table.

Why curate instead of re-run: every per-corpus DUSTED/binding number here is already
validated in BENCHMARK; re-running 5 corpora through heterogeneous loaders would re-derive
numbers we have. Re-running a single corpus to refresh a field is a one-line edit here.

Read: the taxonomy lands each reference corpus on its validated type — Bengalese (discrete,
GT-pure, strongly recurrent) > bat (graded, emitter-signatured, socially-grounded) > marmoset
(sparse, unstable) ; rook (individually-stereotyped, atomic); LibriSpeech (human lexical
anchor — recurrent terms are NOT words). Call-type/context associations are ACOUSTIC, not
validated meaning (``semantic_validation`` stays ``none``).

Run:  python scripts/repertoire_taxonomy.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics.repertoire_taxonomy import (  # noqa: E402
    RepertoireProfile,
)

# Curated per-corpus profiles. Every number is validated in BENCHMARK; ``source`` cites the
# section. None = that diagnostic was not computed for this corpus (the classifier handles it).
PROFILES = [
    RepertoireProfile(
        corpus="bengalese_finch",
        # §22 cross-corpus DUSTED (uniform thr=0.15, HAC), 3 birds × 150 bouts
        family_entropy=0.85,
        top_family_fraction=0.12,
        null_ratio=2.67,
        bootstrap_stability=0.509,
        n_motif_families=457,
        coverage=0.480,
        emitter_v=None,
        context_v=None,
        notes="GT syllable purity 0.788 (the only GT-validated DUSTED recovery); bird-ID not "
        "computed as token-V (syllable is the labeled unit).",
        source="§22 DUSTED; §12 purity",
    ),
    RepertoireProfile(
        corpus="egyptian_fruit_bat",
        # §22 DUSTED (300 bouts); §13 Emitter/Context V; §9 social-lift (D=4.3×≈G=4.7×)
        family_entropy=0.87,
        top_family_fraction=0.06,
        null_ratio=1.36,
        bootstrap_stability=0.394,
        n_motif_families=306,
        coverage=0.153,
        emitter_v=0.446,
        context_v=0.167,
        social_lift=1.74,
        notes="Graded/emitter-signatured bouts; DTW over-merging risk. Context_binding "
        "significant within-Emitter (§18).",
        source="§22 DUSTED; §13 Emitter/Context V; §9 social-lift",
    ),
    RepertoireProfile(
        corpus="infant_marmoset",
        # §22 DUSTED (1500 calls); call-type V from track_b (§18). No individual-ID labels.
        family_entropy=0.89,
        top_family_fraction=0.06,
        null_ratio=1.55,
        bootstrap_stability=0.229,
        n_motif_families=139,
        coverage=0.028,
        emitter_v=None,
        context_v=0.599,
        notes="Whole-call fp recovers call type (V 0.599) but DUSTED sequence motifs are "
        "sparse/unstable (cov 0.028) — one-unit repeats. No individual-ID labels ⇒ "
        "turn_structure not_evaluable.",
        source="§22 DUSTED; §18/track_b call-type V",
    ),
    RepertoireProfile(
        corpus="rook",
        # Atomic whole calls; no uniform-threshold §22 DUSTED. §23 Source (individual-ID) V,
        # social_other (context) ≈ chance.
        family_entropy=None,
        top_family_fraction=None,
        null_ratio=None,
        bootstrap_stability=None,
        n_motif_families=None,
        coverage=None,
        emitter_v=0.31,
        context_v=0.04,
        notes="Atomic whole calls; identity (Source) dominates context (social_other ≈ chance). "
        "Fine-Event retrieval below chance across callers (§26) — caller-bound.",
        source="§23 Source/social_other V; §26 cross-caller",
    ),
    RepertoireProfile(
        corpus="librispeech_human",
        is_human=True,
        notes="Human word-structure anchor. Cross-speaker same-word retrieval 5–7× chance (§25); "
        "recurrent acoustic terms are NOT decoded words (semantic ceiling, §29).",
        source="§25 cross-speaker; §29 word-likeness",
    ),
]


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main():
    rows = [p.as_dict() for p in PROFILES]
    cols = [
        "corpus",
        "repertoire_type",
        "secondary_flags",
        "null_ratio",
        "bootstrap_stability",
        "coverage",
        "n_motif_families",
        "emitter_v",
        "context_v",
        "social_lift",
        "family_entropy",
        "top_family_fraction",
    ]
    print(
        f"\n{'=' * 132}\nCROSS-SPECIES REPERTOIRE TAXONOMY (Architecture v1.0 Step 3)\n{'=' * 132}"
    )
    print("  " + "  ".join(f"{c[:16]:>16}" for c in cols))
    print("  " + "-" * 128)
    for r in rows:
        r2 = dict(r)
        r2["secondary_flags"] = ",".join(r["secondary_flags"]) or "-"
        print("  " + "  ".join(f"{_fmt(r2.get(c))[:16]:>16}" for c in cols))
    print("\n  Descriptions + provenance:")
    for p in PROFILES:
        flags = f" [{','.join(p.secondary_flags)}]" if p.secondary_flags else ""
        print(f"    {p.corpus}: {p.repertoire_type}{flags} — {p.description}")
        print(f"      source: {p.source}")
        if p.notes:
            print(f"      note:   {p.notes}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "repertoire_taxonomy.csv", "w", newline="") as fh:
        # flatten secondary_flags for CSV
        flat = []
        for r in rows:
            d = dict(r)
            d["secondary_flags"] = ";".join(d["secondary_flags"])
            flat.append(d)
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    with open(out_dir / "repertoire_taxonomy.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nsaved outputs/repertoire_taxonomy.{{csv,json}} ({len(rows)} corpora)")
    print("\nREPERTOIRE TAXONOMY DONE")


if __name__ == "__main__":
    main()
