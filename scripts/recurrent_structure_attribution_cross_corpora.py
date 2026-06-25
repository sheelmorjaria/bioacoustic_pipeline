#!/usr/bin/env python3
"""Recurrent-structure attribution across corpora — within-individual vs across-group.

Generalises §41 (zebra finch) to test whether the ~50/50 within-vs-across split for recurrent
bigrams holds across repertoires. The attribution is corpus-agnostic: given (per-recording
unit sequences, individual labels) it computes Herfindahl concentration of each recurrent
bigram over individuals vs a weighted-random null. WITHIN = p<0.05 & top1>=0.5.

Corpora (all annotation/text-only — no audio needed for the unit SEQUENCE):
  * zebra_finch — call-type sequences per session (filename); individual = bird name (21)
  * marmoset    — call-type sequences per session (timestamped Annotations.tsv); ind = parent_name
  * bengalese   — GT syllable-label sequences per bout (Annotation.xml); ind = Bird id (0..10)
  * librispeech — word sequences per utterance (.trans.txt); ind = speaker id (human anchor)

Prediction from §36/§41: recurrent bigrams should be substantially WITHIN-individual for the
animal repertoires (entangled motor signatures) but more ACROSS for human speech (speaker-
invariant words). Ceiling-bound: units are acoustic/lexical categories, NOT meaning.

Run:  python scripts/recurrent_structure_attribution_cross_corpora.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ZF_DATA = Path("/mnt/c/Users/sheel/Desktop/data/zebrafinch")
MAR_VOC = Path("/home/sheel/birdsong_analysis/data/Vocalizations")
MAR_ANNOT = MAR_VOC / "Annotations.tsv"
LS_DATA = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
MAR_TYPES = ("Phee", "Twitter", "Trill", "Tsik", "Seep")


# ---------- attribution core (copied from §41 — corpus-agnostic) ----------
def _herfindahl(counts):
    n = sum(counts)
    return sum((c / n) ** 2 for c in counts) if n else 0.0


def attribute(occ_by_ind, ind_counts, n_perm, seed):
    inds = list(ind_counts)
    w = np.array([ind_counts[i] for i in inds], dtype=float)
    w = w / w.sum()
    obs = [occ_by_ind.get(i, 0) for i in inds]
    n = sum(obs)
    if n == 0:
        return None
    h = _herfindahl(obs)
    top1 = max(obs) / n
    rng = np.random.RandomState(seed)
    null = np.array([_herfindahl(rng.multinomial(n, w)) for _ in range(n_perm)])
    mu, sd = float(null.mean()), float(null.std() + 1e-12)
    return {
        "n": int(n),
        "H": round(h, 3),
        "top1": round(top1, 3),
        "H_null": round(mu, 3),
        "z": round((h - mu) / sd, 2),
        "p": round(float((null >= h).mean()), 4),
        "top_ind": max(occ_by_ind, key=occ_by_ind.get),
        "within": bool((null >= h).mean() < 0.05 and top1 >= 0.5),
    }


def attribute_bigrams(seqs, inds, min_count, n_perm, seed):
    ind_counts = Counter(inds)
    bi = defaultdict(lambda: defaultdict(int))
    for seq, ind in zip(seqs, inds):
        for a, b in zip(seq[:-1], seq[1:]):
            bi[(a, b)][ind] += 1
    rows = []
    for (a, b), occ in bi.items():
        if sum(occ.values()) < min_count:
            continue
        r = attribute(occ, ind_counts, n_perm, seed)
        if r:
            r["bigram"] = f"{a}->{b}"
            rows.append(r)
    rows.sort(key=lambda r: -r["H"])
    return rows


# ---------- per-corpus loaders → (seqs, inds) ----------
def load_zebra_finch():
    import compositional_zebrafinch as zf  # noqa: PLC0415

    sessions = zf.parse_filenames(["adult"])
    seqs, inds = [], []
    for s in sessions.values():
        if len(s["seq"]) >= 3:
            seqs.append(s["seq"])
            inds.append(s["ind"])
    return seqs, inds


def load_marmoset():
    by_sess = defaultdict(list)
    with MAR_ANNOT.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            lab = (row.get("label") or "").strip()
            parent = (row.get("parent_name") or "").strip()
            if lab not in MAR_TYPES or not parent:
                continue
            h, mi, s, ms = (
                row.get("hour"),
                row.get("minute"),
                row.get("second"),
                row.get("millisecond"),
            )
            if not (h and mi and s):
                continue
            try:
                t = (
                    int(float(h)) * 3600
                    + int(float(mi)) * 60
                    + float(s)
                    + (float(ms) / 1000.0 if ms else 0.0)
                )
            except (ValueError, TypeError):
                continue
            by_sess[parent].append((t, lab))
    seqs, inds = [], []
    for parent, calls in by_sess.items():
        calls.sort()
        seq = [lab for _, lab in calls]
        if len(seq) >= 3:
            seqs.append(seq)
            inds.append(parent)
    return seqs, inds


def load_bengalese(max_birds=11):
    import dusted_bengalese as db  # noqa: PLC0415

    seqs, inds = [], []
    for bird in range(max_birds):
        xml = db.DATA / f"Bird{bird}" / f"Bird{bird}" / "Annotation.xml"
        if not xml.exists():
            continue
        try:
            bouts = db.parse_annotation(xml)
        except Exception:  # noqa: BLE001
            continue
        for _wav, notes in bouts:
            seq = [str(label) for _s, _l, label in notes]
            if len(seq) >= 3:
                seqs.append(seq)
                inds.append(f"Bird{bird}")
    return seqs, inds


def load_librispeech(n_utts=600):
    seqs, inds = [], []
    cnt = 0
    for tf in sorted(LS_DATA.rglob("*.trans.txt")):
        spk = tf.parent.name
        for line in tf.read_text().splitlines():
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            words = parts[1].split()
            if len(words) >= 3:
                seqs.append(words)
                inds.append(spk)
                cnt += 1
        if cnt >= n_utts:
            break
    return seqs, inds


LOADERS = {
    "zebra_finch": load_zebra_finch,
    "marmoset": load_marmoset,
    "bengalese": load_bengalese,
    "librispeech": load_librispeech,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="all")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = list(LOADERS) if args.corpora == "all" else [c.strip() for c in args.corpora.split(",")]
    summary = {}
    print("=" * 98)
    print("RECURRENT-STRUCTURE ATTRIBUTION ACROSS CORPORA — within-individual vs across-group")
    print("=" * 98)
    print(
        f"  {'corpus':12} {'n_rec':>6} {'n_ind':>6} {'#bigrams':>9} {'#within':>8} {'%within':>8}  top within (bigram | top_ind | top1)"
    )
    print("  " + "-" * 94)
    for name in names:
        seqs, inds = LOADERS[name]()
        if len(seqs) < 10:
            print(f"  {name:12}  (skipped: only {len(seqs)} sequences)")
            continue
        rows = attribute_bigrams(seqs, inds, args.min_count, args.n_perm, args.seed)
        within = [r for r in rows if r["within"]]
        n_bi = len(rows)
        pct = 100 * len(within) / max(n_bi, 1)
        summary[name] = {
            "n_recordings": len(seqs),
            "n_individuals": len(set(inds)),
            "n_bigrams": n_bi,
            "n_within": len(within),
            "pct_within": round(pct, 1),
            "top_within": [
                {
                    "bigram": r["bigram"],
                    "ind": r["top_ind"],
                    "top1": r["top1"],
                    "H": r["H"],
                    "p": r["p"],
                }
                for r in within[:5]
            ],
            "top_across": [
                {"bigram": r["bigram"], "top1": r["top1"], "H": r["H"]}
                for r in [r for r in rows if not r["within"]][:5]
            ],
        }
        topw = within[0] if within else None
        ex = f"{topw['bigram']} | {topw['top_ind']} | {topw['top1']:.2f}" if topw else "—"
        print(
            f"  {name:12} {len(seqs):>6} {len(set(inds)):>6} {n_bi:>9} {len(within):>8} "
            f"{pct:>7.0f}%  {ex}"
        )

    print("\n  Read: WITHIN = recurrent bigram concentrated in one individual beyond the null")
    print("  (p<0.05 & top1>=0.5); ACROSS = shared species-level structure.")
    print("  §36/§41 prediction: animal repertoires → many WITHIN (entangled motor signatures);")
    print("  human speech → more ACROSS (speaker-invariant words).")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    with open(out / "recurrent_structure_attribution_cross_corpora.json", "w") as fh:
        json.dump(
            {
                "summary": summary,
                "min_count": args.min_count,
                "framing": "Within-individual vs across-group attribution of recurrent "
                "bigrams across corpora. NOT meaning.",
            },
            fh,
            indent=2,
        )
    print("\nsaved outputs/recurrent_structure_attribution_cross_corpora.json")
    print("CROSS-CORPUS ATTRIBUTION DONE")


if __name__ == "__main__":
    main()
