#!/usr/bin/env python3
"""Individual-centric vs group-level recurrent-unit discovery (Entanglement-Ceiling test).

§36 (Entanglement Ceiling), §26 (rook cross-caller retrieval below chance), and §27
(consensus-template failure) all imply that caller identity is non-separably entangled with
call form. CONSEQUENCE for discovery: searching for recurrent acoustic units / motifs at the
GROUP level pools across different vocal apparatuses, and the entangled caller signature
contaminates the codebook. The recommendation: **isolate individual repertoires first**.

This script QUANTIFIES that recommendation on marmoset (the canonical entanglement corpus:
many callers, 5 call-type labels). It discovers recurrent acoustic units via fingerprint
k-means (the "codebook" = recurrent units) three ways and measures call-FORM recovery
(Cramér's V of token ↔ call-type):

  * GROUP      — one shared codebook over all callers pooled (what §11/§39 effectively did).
  * PER-CALLER — a separate codebook per caller (caller signature held fixed).
  * CROSS      — caller A's codebook applied to caller B's calls (transfer / generalisation).

ENTANGLEMENT PREDICTION: per-caller > group > cross. If per-caller recovers call-form MORE
cleanly than group, isolating the individual sharpens discovery (the group codebook is
contaminated by the entangled caller signature); and cross-caller transfer being poor confirms
units don't generalise across vocal apparatuses (§26).

Also a direct within-vs-across NN check: is a call's nearest neighbour more often the same
CALLER or the same call TYPE? (caller dominance ⇒ entanglement).

Firewall-clean: all-species acoustic fingerprint. Ceiling-bound: call types are acoustic
categories, NOT meaning.

Run:  python scripts/individual_vs_group_discovery.py --n-per-type 500 --min-calls-per-caller 20

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VOC_DIR = Path("/home/sheel/birdsong_analysis/data/Vocalizations")
ANNOT = VOC_DIR / "Annotations.tsv"
CALL_TYPES = ("Phee", "Twitter", "Trill", "Tsik", "Seep")
SR = 96000


def _load_index():
    idx = {}
    with ANNOT.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            fn = (row.get("file_name") or "").strip()
            if not fn:
                continue
            idx[fn] = (row.get("label", "").strip(), row.get("parent_name", "").strip())
    return idx


def sample_calls(args, rng):
    """Balanced over call types; callers (parent_name) with >= min_calls, kept per caller."""
    idx = _load_index()
    by_caller = defaultdict(list)
    for fn, (lab, parent) in idx.items():
        if lab in CALL_TYPES and parent:
            by_caller[parent].append((lab, fn))
    eligible = {c for c, lst in by_caller.items() if len(lst) >= args.min_calls_per_caller}
    by_type = defaultdict(list)
    for caller in eligible:
        for lab, fn in by_caller[caller]:
            by_type[lab].append((caller, fn))
    chosen = []
    for ct in CALL_TYPES:
        pool = by_type.get(ct, [])
        rng.shuffle(pool)
        chosen.extend((ct, caller, fn) for caller, fn in pool[: args.n_per_type])
    rng.shuffle(chosen)
    return chosen, eligible


def _v(tokens, labels, n_perm=200, seed=0):
    """Cramér's V between token ids and labels via association_test."""
    from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: PLC0415

    if len(set(tokens)) < 2 or len(set(labels)) < 2:
        return 0.0, 1.0
    r = association_test(
        [[int(t)] for t in tokens], list(labels), statistic="nmi", n_permutations=n_perm, seed=seed
    ).as_dict()
    return r["cramers_v"], r["p_value"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=500)
    ap.add_argument("--min-calls-per-caller", type=int, default=20)
    ap.add_argument("--k-group", type=int, default=24)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    fw = default_firewall()
    fw.assert_allowed("acoustic_fingerprint", Use.TOKENIZE)
    fw.assert_cross_species("acoustic_fingerprint", "marmoset")
    print("firewall: acoustic_fingerprint tokenize legal for marmoset")

    rng = np.random.RandomState(args.seed)
    chosen, eligible = sample_calls(args, rng)
    print(
        f"{len(eligible)} eligible callers (>= {args.min_calls_per_caller} calls); "
        f"sampled {len(chosen)} calls balanced over {CALL_TYPES}"
    )

    fps, ctypes, callers = [], [], []
    for i, (ct, caller, fn) in enumerate(chosen):
        path = VOC_DIR / caller / fn
        if not path.exists():
            continue
        try:
            a, sr = load_audio(str(path), target_sr=SR)
        except Exception:  # noqa: BLE001
            continue
        fp = afp.extract(a, sr, species="marmoset").combined
        if not np.any(fp):
            continue
        fps.append(fp)
        ctypes.append(ct)
        callers.append(caller)
    FP = np.asarray(fps, dtype=np.float32)
    ctypes = np.asarray(ctypes)
    callers = np.asarray(callers)
    print(f"fingerprinted {len(FP)} calls | {len(set(callers))} callers | feat {FP.shape[1]}-D")

    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    def km(X, k, seed):
        return (
            MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=4096)
            .fit(StandardScaler().fit_transform(X))
            .labels_
        )

    # ---- 1. GROUP codebook (all callers pooled) ----
    gtok = km(FP, args.k_group, args.seed)
    v_group_ct, p_group = _v(gtok, ctypes, args.n_perm, args.seed)
    v_group_caller, _ = _v(gtok, callers, args.n_perm, args.seed)  # caller leakage

    # ---- 2. PER-CALLER codebook (caller signature held fixed) ----
    per_rows = []
    uniq = list(set(callers))
    for caller in uniq:
        m = callers == caller
        sub_fp = FP[m]
        sub_ct = ctypes[m]
        if len(sub_fp) < 10 or len(set(sub_ct)) < 2:
            continue
        k = min(args.k_group, max(5, len(sub_fp) // 8))
        toks = km(sub_fp, k, args.seed)
        v, p = _v(toks, sub_ct, min(args.n_perm, 100), args.seed)
        per_rows.append({"caller": caller, "n": int(m.sum()), "v": v, "p": p, "k": k})
    v_per_caller = float(np.average([r["v"] for r in per_rows], weights=[r["n"] for r in per_rows]))

    # ---- 3. CROSS-CALLER transfer: caller A codebook applied to caller B ----
    cross = []
    rng2 = np.random.RandomState(args.seed + 1)
    for a_caller in uniq:
        ma = callers == a_caller
        if ma.sum() < 20:
            continue
        ka = min(args.k_group, max(5, int(ma.sum()) // 8))
        centroids = (
            MiniBatchKMeans(n_clusters=ka, random_state=args.seed, n_init="auto", batch_size=4096)
            .fit(StandardScaler().fit_transform(FP[ma]))
            .cluster_centers_
        )
        sa = StandardScaler().fit(FP[ma])  # NOTE: A's scaling — transfer is intentionally imperfect
        # apply to a different caller B
        others = [c for c in uniq if c != a_caller and (callers == c).sum() >= 20]
        for b_caller in rng2.choice(others, size=min(2, len(others)), replace=False):
            mb = callers == b_caller
            xb = sa.transform(FP[mb])
            btok = np.argmin(((xb[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
            v, _ = _v(btok, ctypes[mb], min(args.n_perm, 100), args.seed)
            cross.append(v)
    v_cross = float(np.mean(cross)) if cross else float("nan")

    # ---- 4. within-vs-across NN: same-caller vs same-calltype dominance ----

    FN = FP / (np.linalg.norm(FP, axis=1, keepdims=True) + 1e-12)
    sim = FN @ FN.T
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(axis=1)
    same_caller = float(np.mean(callers == callers[nn]))
    same_type = float(np.mean(ctypes == ctypes[nn]))

    # ---- report ----
    print("\n" + "=" * 92)
    print("INDIVIDUAL-CENTRIC vs GROUP DISCOVERY — recurrent-unit codebook (marmoset)")
    print("=" * 92)
    print(f"  n_calls={len(FP)}  n_callers={len(uniq)}  k_group={args.k_group}")
    print(f"  {'discovery mode':24} {'V(token,calltype)':>20} {'note'}")
    print("  " + "-" * 88)
    print(f"  {'GROUP (pooled codebook)':24} {v_group_ct:>20.3f}   (p={p_group:.3f})")
    print(
        f"  {'PER-CALLER (per-bird km)':24} {v_per_caller:>20.3f}   (mean over {len(per_rows)} callers)"
    )
    print(
        f"  {'CROSS-CALLER transfer':24} {v_cross:>20.3f}   (A codebook -> B calls, n={len(cross)})"
    )
    print(f"\n  Group codebook caller-leakage: V(token, caller) = {v_group_caller:.3f}")
    print(
        f"  NN check: P(nearest-neighbor same CALLER)={same_caller:.3f}  "
        f"vs same call-TYPE={same_type:.3f}"
    )
    print("\n  Verdict (entanglement prediction: per-caller > group > cross):")
    order = v_per_caller > v_group_ct > v_cross
    print(
        f"    per-caller ({v_per_caller:.3f}) {'>' if v_per_caller > v_group_ct else '<='} "
        f"group ({v_group_ct:.3f}) {'>' if v_group_ct > v_cross else '<='} cross ({v_cross:.3f})"
    )
    print(
        f"    => prediction {'CONFIRMED' if order else 'PARTIAL/NOT confirmed'}: "
        f"isolating the individual {'sharpens' if v_per_caller > v_group_ct else 'does not sharpen'} "
        f"call-form recovery; cross-caller transfer {'is poor' if v_cross < v_group_ct else 'transfers'}."
    )
    print(
        f"    NN: caller signature {'DOMINATES' if same_caller > same_type else 'does not dominate'} "
        f"call-type structure (entanglement {'confirmed' if same_caller > same_type else 'not confirmed'})."
    )
    print("\n  Methodological read: recurrent-unit / motif discovery should be done PER-INDIVIDUAL")
    print("  first (caller signature held fixed), then compared across the group for species-level")
    print("  invariants. Group-level codebooks are contaminated by the entangled caller signature")
    print("  (§36). Ceiling-bound: call types are acoustic categories, NOT meaning.")

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    payload = {
        "n_calls": int(len(FP)),
        "n_callers": int(len(uniq)),
        "k_group": args.k_group,
        "v_group_calltype": v_group_ct,
        "p_group": p_group,
        "v_group_caller": v_group_caller,
        "v_per_caller": v_per_caller,
        "n_per_caller_callers": len(per_rows),
        "v_cross": v_cross,
        "n_cross": len(cross),
        "nn_same_caller": same_caller,
        "nn_same_calltype": same_type,
        "prediction_confirmed": bool(order),
        "framing": "Per-individual discovery sharpens call-form recovery (entanglement). NOT meaning.",
    }
    with open(out / "individual_vs_group_discovery.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nsaved outputs/individual_vs_group_discovery.json")
    print("INDIVIDUAL VS GROUP DISCOVERY DONE")


if __name__ == "__main__":
    main()
