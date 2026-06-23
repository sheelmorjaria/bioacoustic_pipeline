#!/usr/bin/env python3
"""RookID Event/Comment enrichment: whole-call fp vs DUSTED motifs × 6 targets.

The RookID TSVs carry two under-used columns: `Event` (corvid ethogram code, 561
distinct) and `Comment` (free-text social/quality notes, 14% of rows). The existing
scripts parse only Source/Start/End and drop Comment. This script keeps BOTH and
parses them:

  Event   → is_sing (62.5% of rows are song) / activity (sing|nest_building|call) /
            song_variant (the sing-prefix call-type within song)
  Comment → social_other (a named partner rook mentioned in the comment, else "none")
            + overlap flag (Pls / in_Pls / part_of_bout) + quality flag (disrupted/…)

Then two per-call representations are scored against 6 targets:
  whole-call fingerprint (44-D, continuous) — k-means token → Cramér's V / NMI vs
    target + a date-held-out linear probe.
  DUSTED motif (discrete) — one DTW discovery over NBD sub-call units; each call's
    token = the DTW motif family whose signature_pattern (modal cluster-tuple) is a
    contiguous subsequence of the call's cluster list (-1 if none).

Targets: Source (excl Pls/Inc), activity, sing/non-sing, song_variant (sing subset),
social_other, date (recording — a confound negative control).

Honest framing: sing/non-sing & activity are acoustic categories (recoverable from
acoustics by construction); social_other is sparse association (NOT "about" the rook);
date is a leakage guardrail. Discovered motifs are recurrent acoustic structure,
NOT words — `semantic_validation` stays `none`.

Run:  python scripts/rookid_event_comment.py --max-calls 3000

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402

import rookid_individual_id as RI  # noqa: E402  (DATA, SR, seek_audio)

K_SUB = 24  # sub-call unit k-means (matches rookid_dusted)
K_FP = 15  # whole-call fp k-means token
NAMED_ROOKS = [
    "Balbo",
    "Kafka",
    "Brain",
    "Jolene",
    "Merlin",
    "Feisty",
    "Gigi",
    "Elie",
    "Bashir",
    "Tom",
    "Pomme",
    "Osiris",
    "Braad",
    "Siobhan",
    "Jonas",
]
QUALITY_BAD = ["disrupted", "faint", "unclear", "noise", "impulse", "remove", "interval_removable"]
OVERLAP_WORDS = ["part_of_bout", "part_of_next_bout", "concatenated"]
SONG_MODIFIERS = ["_sing", "_porte", "_centre", "_fond", "_micro", "_("]
_ROOK_RE = {r: re.compile(r"\b" + re.escape(r) + r"\b", re.IGNORECASE) for r in NAMED_ROOKS}


def song_variant(event: str) -> str:
    """Sing-prefix call-type (e.g. agw_sing→agw, apt_porte_sing→apt); '' if non-sing."""
    if "sing" not in event.lower():
        return ""
    low = event.lower()
    cut = len(event)
    for m in SONG_MODIFIERS:
        i = low.find(m)
        if i >= 0 and i < cut:
            cut = i
    return event[:cut] or event


def social_other(comment: str, source: str) -> str:
    """Named partner rook mentioned in comment (excluding the source itself); else 'none'."""
    if not comment:
        return "none"
    for r in NAMED_ROOKS:
        if r.lower() == source.lower():
            continue
        if _ROOK_RE[r].search(comment):
            return r
    return "none"


def parse_enriched(max_calls):
    """All TSV rows → enriched call dicts (keeps Comment + Pls/Inc, unlike RI.parse_tsvs)."""
    calls = []
    for tsv in sorted(RI.DATA.glob("*.tsv")):
        stem = tsv.stem
        date = stem[:8]
        with open(tsv) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                src = (row.get("Source") or "").strip()
                ev = (row.get("Event") or "").strip()
                cm = (row.get("Comment") or "").strip()
                try:
                    s = float(row["Start"])
                    e = float(row["End"])
                except (ValueError, KeyError):
                    continue
                if e - s < 0.08 or e - s > 3.0:
                    continue
                if not src or not ev:
                    continue
                low_cm = cm.lower()
                is_sing = "sing" in ev.lower()
                calls.append(
                    dict(
                        stem=stem,
                        source=src,
                        event=ev,
                        comment=cm,
                        is_sing=is_sing,
                        activity=(
                            "sing"
                            if is_sing
                            else "nest_building"
                            if "nest_building" in ev.lower()
                            else "call"
                        ),
                        song_variant=song_variant(ev),
                        social_other=social_other(cm, src),
                        overlap=(
                            src == "Pls"
                            or "in_pls" in ev.lower()
                            or any(w in low_cm for w in OVERLAP_WORDS)
                        ),
                        quality_ok=not any(w in low_cm for w in QUALITY_BAD),
                        date=date,
                        start=s,
                        end=e,
                    )
                )
                if len(calls) >= max_calls:
                    return calls
    return calls


def build_recs(calls):
    """Per call: whole-call fp (44-D) + NBD sub-call unit fingerprints."""
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="").combined  # noqa: E731
    recs = []
    for ci, c in enumerate(calls):
        wav = RI.DATA / f"{c['stem']}.wav"
        if not wav.exists():
            continue
        audio = RI.seek_audio(wav, c["start"], c["end"])
        if audio is None or len(audio) < int(0.03 * RI.SR):
            continue
        whole = np.asarray(fp_fn(audio, RI.SR), dtype=np.float32)
        if not np.any(whole):
            continue
        sig = compute_signals(audio, RI.SR, hop_ms=10.0)
        props = generate_proposals(sig, floor_energy=0.04, floor_flux=0.02, debounce_ms=25.0)
        sub_fps = []
        for p in props:
            si = max(0, int(p.time_ms / 1000 * RI.SR) - int(0.02 * RI.SR))
            sj = si + int(0.04 * RI.SR)
            seg = audio[si:sj]
            if len(seg) >= int(0.012 * RI.SR):
                v = np.asarray(fp_fn(seg, RI.SR), dtype=np.float32)
                if np.any(v):
                    sub_fps.append(v)
        recs.append({**c, "ci": ci, "whole": whole, "sub_fps": sub_fps})
        if (ci + 1) % 500 == 0:
            print(f"  {ci + 1}/{len(calls)} → {len(recs)} calls", flush=True)
    return recs


def kmeans_tokens(vecs, k, seed=0):
    """Global k-means → per-vector cluster id."""
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    X = np.asarray(vecs, dtype=np.float32)
    scaler = StandardScaler().fit(X)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=4096).fit(
        scaler.transform(X)
    )
    sv = scaler.transform(X)
    return np.argmin(((sv[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2), axis=1)


def _contains(clusters, sig):
    """Is tuple `sig` a contiguous subsequence of list `clusters`?"""
    n, m = len(clusters), len(sig)
    if m == 0 or m > n:
        return False
    for i in range(n - m + 1):
        if tuple(clusters[i : i + m]) == sig:
            return True
    return False


def motif_token_per_call(recs, sigs_by_support):
    """Per-call dominant DTW motif family by signature-presence; -1 if none contained.

    ``sigs_by_support``: list of (family_idx, signature_tuple) sorted by descending
    support. Uses each motif's stored signature_pattern (modal cluster-tuple) — a
    faithful, cheap per-call assignment derivable from one discovery (SubsequenceMotif
    does not store member-call lists, so signature-presence is the accessible proxy).
    """
    toks = []
    for r in recs:
        cl = r["clusters"]
        assigned = -1
        for i, sig in sigs_by_support:
            if _contains(cl, sig):
                assigned = i
                break
        toks.append(assigned)
    return toks


def cramers_v(tok, lab):
    from scipy.stats import chi2_contingency  # noqa: PLC0415

    tok = np.asarray(tok)
    lab = np.asarray(lab)
    ut = sorted(set(tok.tolist()))
    ul = sorted(set(lab.tolist()))
    if len(ut) < 2 or len(ul) < 2:
        return 0.0
    table = np.zeros((len(ut), len(ul)))
    for t, lll in zip(tok, lab):
        table[ut.index(t), ul.index(lll)] += 1
    try:
        chi2, _, _, _ = chi2_contingency(table)
        return min(float(np.sqrt(chi2 / (table.sum() * (min(len(ut), len(ul)) - 1)))), 1.0)
    except Exception:  # noqa: BLE001
        return 0.0


def nmi(tok, lab):
    from sklearn.metrics import normalized_mutual_info_score  # noqa: PLC0415

    return float(normalized_mutual_info_score(np.asarray(lab), np.asarray(tok)))


def fp_probe_holdout(recs, target_vals, seed=0):
    """Date-held-out linear probe (whole-call fp → target), macro-F1."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    import random  # noqa: PLC0415

    dates = sorted({r["date"] for r in recs})
    rng = random.Random(seed)
    rng.shuffle(dates)
    split = int(0.7 * len(dates))
    train_dates = set(dates[:split])
    tr = [i for i, r in enumerate(recs) if r["date"] in train_dates]
    te = [i for i, r in enumerate(recs) if r["date"] not in train_dates]
    common = set(target_vals[i] for i in tr) & set(target_vals[i] for i in te)
    tr = [i for i in tr if target_vals[i] in common]
    te = [i for i in te if target_vals[i] in common]
    if len(tr) < 20 or len(te) < 10 or len(common) < 2:
        return None
    Xtr = np.asarray([recs[i]["whole"] for i in tr], dtype=np.float32)
    Xte = np.asarray([recs[i]["whole"] for i in te], dtype=np.float32)
    ytr = [target_vals[i] for i in tr]
    yte = [target_vals[i] for i in te]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0).fit(
        sc.transform(Xtr), ytr
    )
    return f1_score(yte, clf.predict(sc.transform(Xte)), average="macro")


def main():
    ap = argparse.ArgumentParser(description="RookID Event/Comment: fp vs DUSTED motifs × targets")
    ap.add_argument("--max-calls", type=int, default=3000)
    ap.add_argument("--k-sub", type=int, default=K_SUB)
    ap.add_argument("--k-fp", type=int, default=K_FP)
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--dist-threshold", type=float, default=0.15)
    ap.add_argument("--centroid-radius", type=float, default=0.20)
    ap.add_argument("--n-null", type=int, default=400)
    ap.add_argument("--n-perm", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("parsing RookID TSVs (Event + Comment kept)...", flush=True)
    calls = parse_enriched(args.max_calls)
    n_sing = sum(1 for c in calls if c["is_sing"])
    n_social = sum(1 for c in calls if c["social_other"] != "none")
    print(
        f"  {len(calls)} calls | sing {n_sing} ({100 * n_sing / max(len(calls), 1):.0f}%) | "
        f"social_other labeled {n_social} ({100 * n_social / max(len(calls), 1):.0f}%) | "
        f"{len({c['source'] for c in calls})} sources, "
        f"{len({c['date'] for c in calls})} dates",
        flush=True,
    )

    print("building whole-call fp + NBD sub-call units...", flush=True)
    recs = build_recs(calls)
    recs = [r for r in recs if len(r["sub_fps"]) >= 2]  # need >=2 sub-units for a motif
    print(f"  {len(recs)} calls with fp + >=2 sub-units", flush=True)

    # ── fp token (global k-means on whole-call fp) ──
    fp_tok = kmeans_tokens([r["whole"] for r in recs], min(args.k_fp, len(recs)), args.seed)
    for r, t in zip(recs, fp_tok):
        r["fp_tok"] = int(t)

    # ── sub-call cluster ids (global k-means) + DUSTED DTW discovery (ONCE) ──
    all_sub = [fp for r in recs for fp in r["sub_fps"]]
    print(f"  {len(all_sub)} sub-call units → k-means k={args.k_sub}...", flush=True)
    sub_tok = kmeans_tokens(all_sub, args.k_sub, args.seed)
    idx = 0
    sm_calls = []
    for ri, r in enumerate(recs):
        n = len(r["sub_fps"])
        cl = sub_tok[idx : idx + n].tolist()
        idx += n
        r["clusters"] = cl
        sm_calls.append(
            sm.CallUnits(
                fingerprints=r["sub_fps"],
                clusters=cl,
                calltype=r["event"],
                caller=r["source"],
                file_id=r["stem"],
                call_idx=ri,
            )
        )

    common = dict(min_n=args.min_n, max_n=args.max_n, min_support=3, min_calls=2)
    dtw = sm.discover_recurrent_subsequences(
        sm_calls, **common, dist_threshold=args.dist_threshold, centroid_radius=args.centroid_radius
    )
    sd = sm.summarize_discovery(dtw, len(sm_calls))
    print(
        f"\nDUSTED DTW corpus: fams={sd['n_motifs']} cov={sd['coverage']:.3f} "
        f"H={sd['family_entropy_norm']:.2f} top={sd['top_family_support_fraction']:.2f} "
        f"src_V={sd['mean_caller_V']:.3f} evt_V={sd['mean_calltype_V']:.3f}",
        flush=True,
    )
    null = sm.discovery_null(
        sm_calls[: args.n_null],
        dist_threshold=args.dist_threshold,
        centroid_radius=args.centroid_radius,
        min_calls=2,
        n_perm=args.n_perm,
        seed=args.seed,
        min_n=args.min_n,
        max_n=args.max_n,
    )
    jac = sm.bootstrap_stability(
        sm_calls[: args.n_null],
        dist_threshold=args.dist_threshold,
        centroid_radius=args.centroid_radius,
        min_calls=2,
        n_boot=args.n_boot,
        top_k=10,
        seed=args.seed,
        min_n=args.min_n,
        max_n=args.max_n,
    )
    print(
        f"  DTW null: obs={null['observed_support']} null={null['null_mean']:.0f}±"
        f"{null['null_std']:.0f} ({null['observed_support'] / max(null['null_mean'], 1):.2f}x) "
        f"p={null['p']:.3f} | bootstrap Jaccard={jac:.3f}",
        flush=True,
    )

    # ── per-call motif token (signature-presence) ──
    sigs_by_support = sorted(
        [(i, tuple(m.signature_pattern)) for i, m in enumerate(dtw) if m.signature_pattern],
        key=lambda kv: -dtw[kv[0]].support,
    )
    m_tok = motif_token_per_call(recs, sigs_by_support)
    for r, t in zip(recs, m_tok):
        r["motif_tok"] = int(t)
    n_in_motif = sum(1 for t in m_tok if t >= 0)
    print(
        f"  motif-token coverage: {n_in_motif}/{len(recs)} calls match a DTW signature", flush=True
    )

    # ── target table ──
    def lab_source(r):
        return r["source"]

    def lab_activity(r):
        return r["activity"]

    def lab_sing(r):
        return "sing" if r["is_sing"] else "call"

    def lab_variant(r):
        return r["song_variant"]

    def lab_social(r):
        return r["social_other"]

    def lab_date(r):
        return r["date"]

    targets = [
        ("Source", lab_source, lambda r: r["source"] not in ("Pls", "Inc", "")),
        ("activity", lab_activity, lambda r: r["quality_ok"]),
        ("sing/non-sing", lab_sing, lambda r: r["quality_ok"]),
        ("song_variant", lab_variant, lambda r: r["is_sing"] and r["song_variant"]),
        ("social_other", lab_social, lambda r: True),
        ("date", lab_date, lambda r: True),
    ]

    print(f"\n{'=' * 92}")
    print(f"ROOKID Event/Comment — whole-call fp vs DUSTED motif × targets (n={len(recs)})")
    print(f"{'=' * 92}")
    print(
        f"  {'target':14} {'n':>5} {'#cls':>4} | {'fp-V':>6} {'fp-NMI':>7} {'fp-probe':>8} "
        f"| {'mot-V':>6} {'mot-NMI':>7}"
    )
    print("  " + "-" * 78)
    for name, lab, keep in targets:
        idxs = [i for i, r in enumerate(recs) if keep(r)]
        labs = [lab(recs[i]) for i in idxs]
        ut = sorted(set(labs))
        if len(ut) < 2 or len(idxs) < 20:
            print(f"  {name:14} {len(idxs):5d} {len(ut):4d} | (insufficient data)")
            continue
        ft = [recs[i]["fp_tok"] for i in idxs]
        mt = [recs[i]["motif_tok"] for i in idxs]
        probe = fp_probe_holdout([recs[i] for i in idxs], labs, args.seed)
        probe_s = f"{probe:.3f}" if probe is not None else "  n/a"
        print(
            f"  {name:14} {len(idxs):5d} {len(ut):4d} | {cramers_v(ft, labs):6.3f} "
            f"{nmi(ft, labs):7.3f} {probe_s:>8} | {cramers_v(mt, labs):6.3f} {nmi(mt, labs):7.3f}"
        )

    print("\n  fp-V/NMI: k-means(k=15) token on whole-call fp vs target.")
    print("  mot-V/NMI: per-call dominant DTW motif family (signature-presence) vs target.")
    print("  fp-probe: date-held-out LogReg macro-F1 (chance ≈ 1/#cls).")
    print("  Source excl Pls/Inc; activity/sing on quality_ok rows; song_variant on sing subset.")
    print(
        "  social_other/date use all rows ('none' is a class; date is a leakage negative control)."
    )
    print("  Motifs are recurrent acoustic structure, NOT words — semantic_validation stays none.")
    print("\nROOKID EVENT/COMMENT DONE")


if __name__ == "__main__":
    main()
