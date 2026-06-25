#!/usr/bin/env python3
"""Compositional-structure scan on zebra finch call sequences (Semantic-Ceiling-safe).

Zebra finch adult corpus (`/mnt/c/Users/sheel/Desktop/data/zebrafinch/adult`). Each wav is one
call; the filename encodes  `<individual>_<date>-<VT>-<rendition>.wav` where VT[:2] is the
canonical call type (Ag/Be/DC/Di/LT/Ne/So/Te/Th/Tu/Wh, 11 categories, per the dataset's
Animal-Cognition 2016 scheme) and the rendition number is the within-session call order. So
calls sharing `<individual>_<date>` form an ORDERED call sequence (a vocal bout).

This is call-TYPE-level sequence structure — richer than §37 (bat sub-call k-means tokens):
here the "units" are biologist-annotated call types, and one individual spans many date-sessions,
so transition structure can be tested for INDIVIDUAL-SPECIFICITY (repeats across a caller's
sessions) — a structural test bat's per-bout context could not support.

STILL ceiling-bound: call types are acoustic categories, NOT meaning. Compositionality (AB
means something A/B do not) is a meaning claim → Track-C-only. This script reports STRUCTURAL
prerequisites only (T0): non-random transitions, positional specialists, PST syntax richness,
and individual-specific transition style. No meaning is asserted.

Reuses frozen modules: structural_alternatives.{probabilistic_suffix_tree, transition_network},
context_binding.association_test. Filename-only — no audio I/O (instant).

KEY ASSUMPTION (flagged): rendition order is treated as temporal call order within the
recording. The dataset notes do not state this explicitly; it is inferred from "rendition
number" + same bird + same date. The mixed-type sample sequences look bout-like, which supports
this, but it is an assumption.

Run:  python scripts/compositional_zebrafinch.py [--age adult|chick|both] [--min-session 3]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = Path("/mnt/c/Users/sheel/Desktop/data/zebrafinch")
CANON = ["Ag", "Be", "DC", "Di", "LT", "Ne", "So", "Te", "Th", "Tu", "Wh"]


def parse_filenames(age_dirs):
    """Return sessions: {session_key: {'ind': str, 'seq': [calltype, ...]}}."""
    pat = re.compile(r"^(?P<sess>[^-]+)-(?P<vt>[A-Za-z]+)(?P<seq>Seq)?-(?P<idx>\d+[a-z]?)\.wav$")
    sessions = {}
    for age in age_dirs:
        d = DATA / age
        if not d.is_dir():
            continue
        for f in sorted(os.listdir(d)):  # noqa: E731  (os imported below)
            if not f.lower().endswith(".wav"):
                continue
            m = pat.match(f)
            if not m:
                continue
            sess = m["sess"]
            ct = m["vt"][:2].capitalize()  # canonical first-2-letter type
            ct = {"So": "So", "Tu": "Tu"}.get(ct, ct)  # pass-through; keep as-is
            # individual = alphabetic bird-name prefix (robust to the dataset's known
            # naming discrepancies: some files are <name><digits> with no '_' before the date/time,
            # e.g. LblBlu2028 = bird "LblBlu" + time 20:28). Letters-only prefix groups a bird's
            # sessions correctly where sess.split('_')[0] would scatter them.
            ind = re.match(r"[A-Za-z]+", sess).group() if re.match(r"[A-Za-z]+", sess) else sess
            sessions.setdefault(sess, {"ind": ind, "calls": []})
            sessions[sess]["calls"].append((m["idx"], ct))
    # order each session by rendition index (numeric then alpha)
    out = {}
    for sess, info in sessions.items():
        ordered = sorted(
            info["calls"], key=lambda t: (int(re.match(r"(\d+)", t[0]).group(1)), t[0])
        )
        out[sess] = {"ind": info["ind"], "seq": [ct for _, ct in ordered]}
    return out


def bigram_transitions(sequences):
    pair, follower = Counter(), Counter()
    total = 0
    for s in sequences:
        for a, b in zip(s[:-1], s[1:]):
            pair[(a, b)] += 1
            follower[b] += 1
            total += 1
    if total == 0:
        return []
    out = []
    for (a, b), c in pair.items():
        a_first = sum(v for (aa, _bb), v in pair.items() if aa == a)
        p_b_given_a = c / a_first
        marg = follower[b] / total
        out.append((a, b, c, p_b_given_a, marg))
    out.sort(key=lambda t: t[3] / max(t[4], 1e-9), reverse=True)
    return out


def positional_specialization(sequences):
    onset, offset, total = Counter(), Counter(), Counter()
    for s in sequences:
        if not s:
            continue
        onset[s[0]] += 1
        offset[s[-1]] += 1
        for t in s:
            total[t] += 1
    rows = []
    for cl, n in total.items():
        if n < 5:
            continue
        rows.append(
            {
                "type": cl,
                "n": n,
                "onset_frac": round(onset[cl] / max(n, 1), 2),
                "offset_frac": round(offset[cl] / max(n, 1), 2),
            }
        )
    rows.sort(key=lambda d: max(d["onset_frac"], d["offset_frac"]), reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age", default="adult", choices=["adult", "chick", "both"])
    ap.add_argument("--min-session", type=int, default=3, help="min calls/session to include")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics.structural_alternatives import (  # noqa: PLC0415
        probabilistic_suffix_tree as pst,
        transition_network as tn,
    )

    age_dirs = ["adult", "chick"] if args.age == "both" else [args.age]
    sessions = parse_filenames(age_dirs)

    seqs = [s["seq"] for s in sessions.values() if len(s["seq"]) >= args.min_session]
    inds = [s["ind"] for s in sessions.values() if len(s["seq"]) >= args.min_session]
    n_calls = sum(len(s) for s in seqs)
    print(
        f"zebra finch ({args.age}): {len(sessions)} sessions, {len(seqs)} with >={args.min_session} calls, "
        f"{n_calls} calls total"
    )
    type_counts = Counter(t for s in seqs for t in s)
    print(f"call-type distribution: {dict(type_counts.most_common())}")

    # ---- T0 structural prerequisites (call-type level) ----
    trans = bigram_transitions(seqs)
    cands = [
        {
            "a": a,
            "b": b,
            "count": c,
            "p_b_given_a": round(pba, 3),
            "marginal_p_b": round(pb, 3),
            "tp_ratio": round(pba / max(pb, 1e-9), 2),
        }
        for a, b, c, pba, pb in trans
        if a != b and c >= 5 and (pba / max(pb, 1e-9)) >= 2.0
    ]
    specs = positional_specialization(seqs)
    # token ids for PST / network
    vocab = {t: i for i, t in enumerate(sorted(type_counts))}
    id_seqs = [[vocab[t] for t in s] for s in seqs]
    pst_sum = pst.fit_pst(id_seqs)
    net = tn.transition_network_metrics(id_seqs)

    # ---- individual-specificity of transition structure ----
    # Does the presence of a bigram in a session predict the individual? (a caller repeats
    # its transition style across its sessions). Test TWO candidates: the top high-TP pair
    # (often a rare, one-bird idiosyncrasy) AND the most-common distinct pair (the robust
    # population signal). Permutation null via association_test.
    ind_counts = Counter(inds)
    multi_inds = {i for i, n in ind_counts.items() if n >= 2}
    keep = [k for k, s in enumerate(seqs) if inds[k] in multi_inds]
    ind_spec = None
    if len(keep) >= 20 and len(multi_inds) >= 3 and trans:
        from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415

        sub_inds = [inds[k] for k in keep]
        # candidate 1: top high-TP (rare). candidate 2: most-common distinct.
        common = max(
            ((a, b, c) for a, b, c, _p, _m in trans if a != b), key=lambda t: t[2], default=None
        )
        tests = []
        if cands:
            tests.append(("top_TP", cands[0]["a"], cands[0]["b"]))
        if common:
            tests.append(("most_common", common[0], common[1]))
        rows = []
        for label, a, b in tests:
            tok = [
                [
                    1
                    if any(seqs[k][j] == a and seqs[k][j + 1] == b for j in range(len(seqs[k]) - 1))
                    else 0
                ]
                for k in keep
            ]
            if len({t[0] for t in tok}) < 2:
                continue
            r = association_test(
                tok, sub_inds, statistic="nmi", n_permutations=args.n_perm, seed=args.seed
            ).as_dict()
            rows.append(
                {"label": label, "candidate": [a, b], "v": r["cramers_v"], "p": r["p_value"]}
            )
        ind_spec = {"n_sessions": len(keep), "n_inds": len(multi_inds), "rows": rows}

    # ---- report ----
    print("\n" + "=" * 88)
    print("ZEBRA FINCH COMPOSITIONAL SCAN (call-type sequences, Semantic-Ceiling-safe, T0 only)")
    print("=" * 88)
    print(
        f"  PST: significant_nodes={pst_sum['n_significant_nodes']} "
        f"max_depth={pst_sum['max_depth_reached']} entropy_rate="
        f"{round(pst_sum['entropy_rate'], 3) if pst_sum['entropy_rate'] is not None else 'n/a'}"
    )
    print(
        f"  transition network: nodes={net.get('n_nodes')} edges={net.get('n_edges')} "
        f"communities={net.get('modularity_communities')} density={net.get('density', 0):.3f}"
    )
    print("\n  Top high-TP distinct call-type bigrams (P(B|A)/P(B) >= 2, count >= 5):")
    print(f"    {'A':>4} {'B':>4} {'count':>6} {'P(B|A)':>8} {'P(B)':>7} {'ratio':>6}")
    for c in cands[:14]:
        print(
            f"    {c['a']:>4} {c['b']:>4} {c['count']:>6} {c['p_b_given_a']:>8.3f} "
            f"{c['marginal_p_b']:>7.3f} {c['tp_ratio']:>6.2f}"
        )
    print("\n  Positional specialists (onset/offset-biased call types):")
    print(f"    {'type':>5} {'n':>6} {'onset_frac':>11} {'offset_frac':>12}")
    for s in specs[:8]:
        print(f"    {s['type']:>5} {s['n']:>6} {s['onset_frac']:>11.2f} {s['offset_frac']:>12.2f}")
    if ind_spec and ind_spec["rows"]:
        print(
            f"\n  Individual-specificity ({ind_spec['n_inds']} birds, {ind_spec['n_sessions']} multi-session sessions):"
        )
        print(f"    {'bigram':>10} {'label':>12} {'V':>7} {'p':>8}")
        for r in ind_spec["rows"]:
            sig = "  *" if r["p"] < 0.05 else ""
            print(
                f"    {'->'.join(r['candidate']):>10} {r['label']:>12} {r['v']:>7.3f} {r['p']:>8.3f}{sig}"
            )
        any_sig = any(r["p"] < 0.05 for r in ind_spec["rows"])
        print(
            f"    => transition structure is {'INDIVIDUAL-SPECIFIC' if any_sig else 'not clearly individual-specific'}"
            f" (* p<0.05). Note: rare-pair V reflects one bird's idiosyncrasy, not a population rule."
        )
    else:
        print("\n  Individual-specificity: skipped (too few multi-session birds or no candidates).")
    print(
        "\n  Framing: these are STRUCTURAL prerequisites (T0) — non-random call-type transitions,"
    )
    print(
        "  positional roles, and (if significant) caller-specific transition style. Whether any AB"
    )
    print(
        "  sequence is compositional (a meaning claim) is UNDECIDABLE by computation -> Track-C playback."
    )
    print("  KEY ASSUMPTION: rendition order treated as temporal call order within the recording.")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    payload = {
        "age": args.age,
        "n_sessions": len(sessions),
        "n_seqs": len(seqs),
        "n_calls": n_calls,
        "type_counts": dict(type_counts.most_common()),
        "pst": pst_sum,
        "transition_network": net,
        "bigram_candidates": cands[:25],
        "positional_specialists": specs[:12],
        "individual_specificity": ind_spec,
        "framing": "Structural prerequisites (T0). NOT compositionality. rendition-order-is-temporal assumption.",
    }
    with open(out / "compositional_zebrafinch.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(out / "compositional_zebrafinch.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["a", "b", "count", "p_b_given_a", "marginal_p_b", "tp_ratio"]
        )
        w.writeheader()
        w.writerows(cands[:60])
    print("\nsaved outputs/compositional_zebrafinch.{json,csv}")
    print("ZEBRA FINCH COMPOSITIONAL SCAN DONE")


if __name__ == "__main__":
    main()
