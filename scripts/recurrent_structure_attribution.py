#!/usr/bin/env python3
"""Attribute recurrent structures to WITHIN-individual vs ACROSS-group origin.

§40 showed coarse call-TYPE recovery is caller-invariant (across-group), while §38 found the
Be↔Lt bigram is one bird's (LblBlu) idiosyncrasy (within-individual). This script generalises
that into a SYSTEMATIC attribution: for every recurrent structure (here, call-type bigrams +
unigrams), partition its occurrences by individual and measure whether it is concentrated in
one bird (WITHIN / idiosyncratic) or spread across many (ACROSS / shared species-level).

METHOD — individual-concentration vs a weighted-random null. For a structure with N
occurrences spread over birds b1..bk with counts n1..nk:
  * Herfindahl  H = Σ (ni/N)²              ∈ [1/k, 1]; high ⇒ concentrated in few birds.
  * top1        = max(ni)/N                 fraction in the single most-common bird.
  * NULL        = H when the structure's occurrences are randomly assigned to sessions
    weighted by each bird's session frequency (permutation). z = (H_obs − mean)/std; perm p.
  * WITHIN-bird  ⇒ concentrated beyond null (p < α and top1 high).
  * ACROSS-group ⇒ spread ≈ null (shared species-level structure).

This answers the question "are the recurrent structures found from individuals or across
groups?" quantitatively, per structure and in aggregate. It is structure-agnostic — the same
attribution applies to DUSTED motif families or k-means token bigrams given (occurrences,
individual labels). Run here on zebra finch (21 birds, call-type sequences; reuses the §38
parser). Ceiling-bound: call types are acoustic categories, NOT meaning.

Run:  python scripts/recurrent_structure_attribution.py --age adult --min-count 5

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import compositional_zebrafinch as zf  # noqa: E402  (reuse the §38 parser)


def _herfindahl(counts):
    n = sum(counts)
    return sum((c / n) ** 2 for c in counts) if n else 0.0


def attribute(struct_occ_by_bird, bird_session_counts, n_perm, seed):
    """One structure -> {n, H, top1, H_null_mean, H_null_std, p, verdict}.

    struct_occ_by_bird: {bird: n_occurrences}. bird_session_counts: {bird: n_sessions} (for
    the weighted-random null). Permutation shuffles the structure's occurrences across birds
    weighted by session frequency."""
    birds = list(bird_session_counts)
    weights = np.array([bird_session_counts[b] for b in birds], dtype=float)
    weights = weights / weights.sum()
    obs_counts = [struct_occ_by_bird.get(b, 0) for b in birds]
    n = sum(obs_counts)
    if n == 0:
        return None
    h_obs = _herfindahl(obs_counts)
    top1 = max(obs_counts) / n
    # null: multinomial draw of n occurrences over birds with `weights`
    rng = np.random.RandomState(seed)
    null_h = []
    for _ in range(n_perm):
        draw = rng.multinomial(n, weights)
        null_h.append(_herfindahl(draw))
    null_h = np.array(null_h)
    mu, sd = float(null_h.mean()), float(null_h.std() + 1e-12)
    z = (h_obs - mu) / sd
    p = float((null_h >= h_obs).mean())
    verdict = "within" if (p < 0.05 and top1 >= 0.5) else "across"
    return {
        "n": int(n),
        "H": round(h_obs, 3),
        "top1": round(top1, 3),
        "H_null": round(mu, 3),
        "z": round(z, 2),
        "p": round(p, 4),
        "top_bird_share": round(top1, 3),
        "verdict": verdict,
    }


def collect_structures(seqs, birds):
    """Return {structure: {bird: count}} for unigrams and bigrams."""
    uni = defaultdict(lambda: defaultdict(int))
    bi = defaultdict(lambda: defaultdict(int))
    for seq, bird in zip(seqs, birds):
        for t in seq:
            uni[t][bird] += 1
        for a, b in zip(seq[:-1], seq[1:]):
            bi[(a, b)][bird] += 1
    return uni, bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age", default="adult", choices=["adult", "chick", "both"])
    ap.add_argument("--min-session", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=5, help="min occurrences for a structure")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    age_dirs = ["adult", "chick"] if args.age == "both" else [args.age]
    sessions = zf.parse_filenames(age_dirs)
    seqs = [s["seq"] for s in sessions.values() if len(s["seq"]) >= args.min_session]
    birds = [s["ind"] for s in sessions.values() if len(s["seq"]) >= args.min_session]
    bird_session_counts = Counter(birds)
    print(
        f"zebra finch ({args.age}): {len(seqs)} sessions, {len(bird_session_counts)} birds, "
        f"{sum(len(s) for s in seqs)} calls"
    )

    uni, bi = collect_structures(seqs, birds)

    def attribute_set(struct_map, label):
        rows = []
        for struct, occ in struct_map.items():
            if sum(occ.values()) < args.min_count:
                continue
            r = attribute(occ, bird_session_counts, args.n_perm, args.seed)
            if r is None:
                continue
            r["structure"] = struct if isinstance(struct, str) else "->".join(struct)
            r["top_bird"] = max(occ, key=occ.get) if isinstance(struct, tuple) else None
            rows.append(r)
        rows.sort(key=lambda r: -r["H"])
        within = [r for r in rows if r["verdict"] == "within"]
        print(
            f"\n  {label}: {len(rows)} recurrent structures (count >= {args.min_count}); "
            f"{len(within)} WITHIN-individual ({100 * len(within) / max(len(rows), 1):.0f}%), "
            f"{len(rows) - len(within)} ACROSS-group"
        )
        return rows, within

    print("\n" + "=" * 96)
    print("RECURRENT-STRUCTURE ATTRIBUTION — within-individual vs across-group (zebra finch)")
    print("=" * 96)
    print("  (Herfindahl H over birds vs weighted-random null; WITHIN = p<0.05 & top1>=0.5)")
    uni_rows, uni_within = attribute_set(uni, "unigrams (call types)")
    bi_rows, bi_within = attribute_set(bi, "bigrams (A->B transitions)")

    print("\n  Top WITHIN-individual bigrams (idiosyncratic — one bird's signature):")
    print(
        f"    {'bigram':>10} {'n':>5} {'H':>6} {'top1':>6} {'H_null':>7} {'z':>6} {'p':>7}  top_bird"
    )
    for r in bi_within[:10]:
        print(
            f"    {r['structure']:>10} {r['n']:>5} {r['H']:>6.3f} {r['top1']:>6.3f} "
            f"{r['H_null']:>7.3f} {r['z']:>6.2f} {r['p']:>7.3f}  {r['top_bird']}"
        )
    print("\n  Top ACROSS-group bigrams (shared species-level structure):")
    for r in [r for r in bi_rows if r["verdict"] == "across"][:8]:
        print(
            f"    {r['structure']:>10} {r['n']:>5} {r['H']:>6.3f} {r['top1']:>6.3f} "
            f"{r['H_null']:>7.3f} {r['z']:>6.2f} {r['p']:>7.3f}"
        )

    print("\n  Unigram attribution (are call types themselves within or across?):")
    for r in uni_rows:
        print(
            f"    {r['structure']:>5} n={r['n']:>4} H={r['H']:.3f} top1={r['top1']:.3f} "
            f"H_null={r['H_null']:.3f} z={r['z']:.2f} p={r['p']:.3f} -> {r['verdict'].upper()}"
        )

    n_bi = len(bi_rows)
    n_bi_within = len(bi_within)
    print("\n  Verdict:")
    print(
        f"    Of {n_bi} recurrent bigrams, {n_bi_within} ({100 * n_bi_within / max(n_bi, 1):.0f}%) are "
        f"WITHIN-individual (concentrated in one bird beyond the null) and "
        f"{n_bi - n_bi_within} ({100 * (n_bi - n_bi_within) / max(n_bi, 1):.0f}%) are ACROSS-group (shared)."
    )
    if n_bi_within > 0:
        print("    => recurrent structure is substantially INDIVIDUAL-SPECIFIC: the pipeline's")
        print(
            "       'recurrent motifs' are often one bird's motor signature, not a shared species"
        )
        print(
            "       rule. This is the discovery-side consequence of the Entanglement Ceiling (§36):"
        )
        print("       validate motifs PER-INDIVIDUAL before claiming species-level structure.")
    else:
        print("    => recurrent structure is SHARED across individuals (species-level), consistent")
        print("       with §40 (coarse call-type structure is caller-invariant).")
    print("  (Generalises: the same attribution applies to DUSTED motifs / token bigrams given")
    print("   occurrences + individual labels.)  Ceiling-bound: not meaning.")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    payload = {
        "age": args.age,
        "n_sessions": len(seqs),
        "n_birds": len(bird_session_counts),
        "n_bigrams": n_bi,
        "n_within": n_bi_within,
        "pct_within": round(100 * n_bi_within / max(n_bi, 1), 1),
        "unigram_attribution": uni_rows,
        "bigram_attribution": bi_rows,
        "method": "Herfindahl over birds vs weighted-random null; within = p<0.05 & top1>=0.5",
        "framing": "Attribute recurrent structure to within-individual vs across-group. NOT meaning.",
    }
    with open(out / "recurrent_structure_attribution.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nsaved outputs/recurrent_structure_attribution.json")
    print("RECURRENT STRUCTURE ATTRIBUTION DONE")


if __name__ == "__main__":
    main()
