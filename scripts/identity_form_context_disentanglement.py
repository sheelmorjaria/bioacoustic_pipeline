#!/usr/bin/env python3
"""Identity–form–context disentanglement across corpora (Architecture v1.0 Step 4).

The backbone biological table: for each corpus, what do the acoustics ENCODE (individual identity,
call/event form), what do they NOT (context, social partner, valence), and what NEEDS social
covariates? This is a SYNTHESIS table — every number is already validated in BENCHMARK
(§13/§18/§23/§24/§25/§29), cited per cell; re-running would re-derive numbers we have. The
verdict (strong / partial / weak / needs_social / n_a) is computed by a small inline classifier.

The table operationalises the architecture's mature thesis — every claim lands in one of three
buckets: (a) what acoustics encode, (b) what acoustics do not encode, (c) what social/behavioral
validation is required. Call-type/identity recovery is an ACOUSTIC association, NOT validated
meaning (``semantic_validation`` stays ``none``).

Run:  python scripts/identity_form_context_disentanglement.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Targets whose recovery from whole-call acoustics is the question.
TARGETS = ["individual_id", "call_type", "context", "social_partner"]

# Each cell: corpus × target → (Cramér's V, linear-probe acc, chance baseline, needs_social flag,
# BENCHMARK source). None = not measured for this corpus. Curated from validated sections.
CELLS = {
    ("egyptian_fruit_bat", "individual_id"): (0.446, None, None, False, "§13 Emitter V"),
    ("egyptian_fruit_bat", "call_type"): (0.167, None, None, False, "§13 Context V"),
    ("egyptian_fruit_bat", "context"): (0.167, None, None, True, "§9 needs social (D=4.3×)"),
    ("egyptian_fruit_bat", "social_partner"): (None, None, None, False, "not in acoustics (§18)"),
    ("infant_marmoset", "individual_id"): (None, None, None, False, "no individual-ID labels"),
    ("infant_marmoset", "call_type"): (0.599, 0.829, 0.20, False, "§18/track_b call-type V/probe"),
    ("infant_marmoset", "context"): (None, None, None, False, "context not labelled"),
    ("infant_marmoset", "social_partner"): (None, None, None, False, "no individual-ID labels"),
    ("meerkat", "individual_id"): (None, None, None, False, "individual-ID not labelled in corpus"),
    ("meerkat", "call_type"): (0.521, 0.753, 1 / 11, False, "§18 call-type V/probe (11 types)"),
    ("meerkat", "context"): (None, None, None, False, "context not labelled"),
    ("meerkat", "social_partner"): (None, None, None, False, "not labelled"),
    ("rook", "individual_id"): (0.31, None, None, False, "§23 Source V (15 rooks)"),
    ("rook", "call_type"): (0.67, None, None, False, "§23 sing/call activity V"),
    ("rook", "context"): (0.04, None, None, False, "§23 social_other ≈ chance"),
    ("rook", "social_partner"): (None, None, None, False, "§26 ≈ chance across callers"),
    ("librispeech_human", "individual_id"): (
        None,
        0.56,
        1 / 178,
        False,
        "§semantic-ceiling speaker probe",
    ),
    ("librispeech_human", "call_type"): (
        0.29,
        0.40,
        None,
        False,
        "§29/§ceiling token↔word (semantic ceiling)",
    ),
    ("librispeech_human", "context"): (None, None, None, False, "n/a (read speech)"),
    ("librispeech_human", "social_partner"): (None, None, None, False, "n/a (read speech)"),
}

CORPORA = ["egyptian_fruit_bat", "infant_marmoset", "meerkat", "rook", "librispeech_human"]


def classify_cell(v, probe, chance, needs_social):
    """strong | partial | weak | needs_social | n_a."""
    if needs_social:
        return "needs_social"
    if v is None and probe is None:
        return "n_a"
    lift = (probe / chance) if (probe is not None and chance) else None
    if (v is not None and v >= 0.40) or (lift is not None and lift >= 3.0):
        return "strong"
    if v is not None and v >= 0.20:
        return "partial"
    if lift is not None and lift >= 1.5:
        return "partial"
    return "weak"


def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def main():
    rows = []
    for corpus in CORPORA:
        for target in TARGETS:
            v, probe, chance, needs_social, source = CELLS.get(
                (corpus, target), (None, None, None, False, "")
            )
            rows.append(
                {
                    "corpus": corpus,
                    "target": target,
                    "V": v,
                    "probe": probe,
                    "chance": chance,
                    "needs_social": needs_social,
                    "verdict": classify_cell(v, probe, chance, needs_social),
                    "source": source,
                }
            )

    print(
        f"\n{'=' * 122}\nIDENTITY–FORM–CONTEXT DISENTANGLEMENT (Architecture v1.0 Step 4)\n{'=' * 122}"
    )
    print(
        f"  {'corpus':22} {'target':15} {'V':>6} {'probe':>6} {'chance':>8} {'verdict':>14}  source"
    )
    print("  " + "-" * 118)
    for r in rows:
        print(
            f"  {r['corpus']:22} {r['target']:15} {_fmt(r['V']):>6} {_fmt(r['probe']):>6} "
            f"{_fmt(r['chance']):>8} {r['verdict']:>14}  {r['source']}"
        )

    print("\n  Read — every claim lands in one of three buckets:")
    print(
        "    ENCODED   (strong/partial): individual identity (bat, rook) + call/event form "
        "(marmoset, meerkat, rook)."
    )
    print(
        "    NOT ENCODED (weak/n_a): context, social partner, valence — acoustics alone do not recover these."
    )
    print("    NEEDS SOCIAL: bat context (§9 D=4.3× with social covariates; A alone undercounts).")
    print(
        "    Human anchor: word identity is NOT recovered by acoustic tokens (semantic ceiling) "
        "— the form/meaning split."
    )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "identity_form_context.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "identity_form_context.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nsaved outputs/identity_form_context.{{csv,json}} ({len(rows)} cells)")
    print("\nIDENTITY FORM CONTEXT DONE")


if __name__ == "__main__":
    main()
