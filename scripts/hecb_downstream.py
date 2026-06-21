#!/usr/bin/env python3
"""HECB-Seg downstream: does typed internal sub-call structure add value?

The segmentation result (2026-06-21) established that PCEN+CRNN is the authoritative
call-event detector and HECB-Seg's role is to ANNOTATE typed internal sub-call boundaries
(it preserves CRNN timing exactly; +793 boundaries across 213 calls on MeerKAT). This
script answers the follow-up: does that internal structure help DOWNSTREAM?

Comparison (per the validated MeerKAT call-type corpus):
  1. Whole CRNN call           — call-type recovery baseline (fingerprint → kmeans → V / probe)
  2. Whole + sub-call pooled   — does an internal-structure summary improve recovery?
  3. Sub-call motif structure  — do within-call sub-call token sequences carry call-type
                                 signal? (sub-call-token ↔ parent-call-type V + per-call LZ)

Uses the cached CRNN + typed classifier from `train_hecb_seg.py` and the production
`segmentation.hierarchical.hecb_decode` decoder (no snapping — CRNN timing preserved).

Usage:
  python3 scripts/hecb_downstream.py --n-clips 1500

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.context_binding import association_test  # noqa: E402
from bioacoustic_pipeline.diagnostics.lempel_ziv import lempel_ziv_complexity  # noqa: E402
from bioacoustic_pipeline.diagnostics.probes import linear_probe  # noqa: E402
from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.segmentation.hierarchical import fit_hecb_priors, hecb_decode  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

# Reuse the cached CRNN + typed classifier + inference helpers from the training script.
import train_hecb_seg as H  # noqa: E402

LBL, WAV, SR = H.LBL, H.WAV, H.SR


def read_calls(h5path):
    """Parse one label .h5 -> [(start_s, end_s, call_type), ...]."""
    with h5py.File(h5path, "r") as h:
        st = np.asarray(h["start_time_lbl"][:]).ravel()
        et = np.asarray(h["end_time_lbl"][:]).ravel()
        lbl = np.asarray(h["lbl"][:]).ravel()
    out = []
    for s, e, lab in zip(st, et, lbl):
        s = float(s)
        if not np.isfinite(s):
            continue
        e = float(e) if np.isfinite(e) else s + 0.08
        if isinstance(lab, bytes):
            ls = lab.decode()
        elif np.isscalar(lab) and np.isfinite(lab):
            ls = str(int(lab))
        else:
            continue
        out.append((s, e, ls))
    return out


def sample_clips(n, seed):
    rng = random.Random(seed)
    files = sorted(glob.glob(str(LBL / "*.h5")))
    rng.shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        calls = read_calls(f)
        if not calls:
            continue
        stem = Path(f).stem
        if (WAV / f"{stem}.wav").exists():
            out.append((stem, calls))
    return out


def match_events_to_calls(events_onset_ms, calls, tol_ms=100.0):
    """Greedy one-to-one match CRNN event onsets → GT calls. Returns {ev_idx: call_type}."""
    used = set()
    matched = {}
    gt_ms = [(s * 1000.0, lab) for s, _e, lab in calls]
    for ei, pon in enumerate(events_onset_ms):
        best, bd = None, tol_ms
        for gi, (g, lab) in enumerate(gt_ms):
            if gi in used:
                continue
            d = abs(pon - g)
            if d < bd:
                bd, best = d, gi
        if best is not None:
            used.add(best)
            matched[ei] = gt_ms[best][1]
    return matched


def kmeans_codebook(vecs, k, seed=0):
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(vecs)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaler.transform(vecs))
    return scaler, km.cluster_centers_


def tokenize(vecs, scaler, cents):
    sv = scaler.transform(vecs)
    return np.argmin(((sv[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1)


def recovery(vecs, labels, k, n_perm=500):
    """k-means tokens → token↔call-type V/nmi/p + linear probe accuracy."""
    scaler, cents = kmeans_codebook(vecs, k)
    toks = tokenize(vecs, scaler, cents)
    binding = association_test(
        [[int(t)] for t in toks], labels.tolist(), statistic="nmi", n_permutations=n_perm, seed=0
    ).as_dict()
    probe = linear_probe(vecs, labels, min_per_class=5).as_dict()
    return {
        "V": binding["cramers_v"],
        "nmi": binding["observed_nmi"],
        "p": binding["p_value"],
        "probe": probe.get("accuracy"),
    }


def main():
    ap = argparse.ArgumentParser(description="HECB-Seg downstream call-type recovery")
    ap.add_argument("--n-clips", type=int, default=1500)
    ap.add_argument("--k", type=int, default=12, help="k-means token count")
    ap.add_argument("--tol-ms", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    # ─── Load cached CRNN + typed classifier ───
    cache = ROOT / "checkpoints" / "hecb"
    crnn = H.CRNNDetector().to(device)
    crnn.load_state_dict(torch.load(cache / "crnn.pt", map_location=device))
    crnn.eval()
    import pickle  # noqa: PLC0415

    with open(cache / "typed.pkl", "rb") as fh:
        _X, _y, clf = pickle.load(fh)
    print("loaded cached CRNN + typed classifier", flush=True)

    # Classifier was trained WITH fingerprint delta (32-D) → inference must match.
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="meerkat").combined  # noqa: E731
    profile = load_species_profiles()["meerkat"]

    clips = sample_clips(args.n_clips, args.seed)
    print(f"{len(clips)} clips with call-type labels", flush=True)

    # ─── Collect whole-call + sub-call fingerprints for matched, labeled calls ───
    whole_vecs, pooled_vecs, labels = [], [], []
    call_sub_groups = []  # (parent_label, [sub_fp,...]) for calls with internal segs
    n_calls_with_internal = 0
    n_internal_total = 0
    fp_dim = None

    all_spans = [[(s, e) for s, e, _l in calls] for _, calls in clips]
    priors = fit_hecb_priors(all_spans)

    for ci, (stem, calls) in enumerate(clips):
        try:
            audio, _ = load_audio(str(WAV / f"{stem}.wav"), target_sr=SR)
        except Exception:
            continue
        proba = H.crnn_inference(crnn, audio, device)
        crnn_events = H.proba_to_events(proba)
        props = H.event_conditioned_proposals(audio, SR, proba, crnn_events, profile, fp_fn)
        typed = [(t, clf.predict_proba(ev.reshape(1, -1))[0]) for t, _p, ev in props]
        events = hecb_decode(crnn_events, typed, priors, file_id=stem)
        matched = match_events_to_calls([e.onset_s * 1000 for e in events], calls, args.tol_ms)

        for ei, ev in enumerate(events):
            if ei not in matched:
                continue
            lab = matched[ei]
            i = int(ev.onset_s * SR)
            j = max(int(ev.offset_s * SR), i + int(0.03 * SR))
            whole_audio = audio[i:j]
            if len(whole_audio) < int(0.015 * SR):
                continue
            wfp = np.asarray(
                afp.extract(whole_audio, SR, species="meerkat").combined, dtype=np.float32
            )
            if not np.any(wfp):
                continue
            if fp_dim is None:
                fp_dim = wfp.shape[0]
            whole_vecs.append(wfp)
            labels.append(lab)

            sfps = []
            for sg in ev.internal_segments:
                si = int(sg.start_s * SR)
                sj = max(int(sg.end_s * SR), si + int(0.015 * SR))
                sseg = audio[si:sj]
                if len(sseg) < int(0.012 * SR):
                    continue
                sfp = np.asarray(
                    afp.extract(sseg, SR, species="meerkat").combined, dtype=np.float32
                )
                if np.any(sfp):
                    sfps.append(sfp)
            if sfps:
                n_calls_with_internal += 1
                n_internal_total += len(sfps)
                mean_sub = np.mean(sfps, axis=0)
                pooled_vecs.append(
                    np.concatenate(
                        [
                            wfp,
                            mean_sub,
                            np.array(
                                [len(sfps), len(sfps) / max(len(ev.internal_segments), 1)],
                                dtype=np.float32,
                            ),
                        ]
                    )
                )
                call_sub_groups.append((lab, sfps))
            else:
                pooled_vecs.append(
                    np.concatenate(
                        [
                            wfp,
                            np.zeros(fp_dim, dtype=np.float32),
                            np.array([0.0, 0.0], dtype=np.float32),
                        ]
                    )
                )

        if (ci + 1) % 200 == 0:
            print(
                f"  {ci + 1}/{len(clips)} clips | {len(whole_vecs)} calls | "
                f"{n_calls_with_internal} with internal",
                flush=True,
            )

    whole_vecs = np.asarray(whole_vecs, dtype=np.float32)
    pooled_vecs = np.asarray(pooled_vecs, dtype=np.float32)
    labels = np.asarray(labels)
    cat_counts = Counter(labels.tolist())
    print(
        f"\nmatched labeled calls: {len(whole_vecs)} | fp dim {fp_dim} | "
        f"call types: {dict(cat_counts)}",
        flush=True,
    )
    print(
        f"calls with internal boundaries: {n_calls_with_internal} ({n_internal_total} sub-segs)",
        flush=True,
    )

    # ─── 1. Whole-call recovery (baseline) ───
    r_whole = recovery(whole_vecs, labels, args.k)
    # ─── 2. Pooled (whole + sub-call summary) recovery ───
    r_pooled = recovery(pooled_vecs, labels, args.k)

    print(f"\n=== Call-type recovery (k={args.k} tokens, MeerKAT) ===")
    print(f"{'representation':24} {'Cramers V':>10} {'NMI':>8} {'p':>10} {'probe acc':>10}")
    print("-" * 66)
    print(
        f"{'whole CRNN call':24} {r_whole['V']:10.3f} {r_whole['nmi']:8.3f} "
        f"{str(r_whole['p']):>10} {str(r_whole['probe']):>10}"
    )
    print(
        f"{'whole + sub-call pool':24} {r_pooled['V']:10.3f} {r_pooled['nmi']:8.3f} "
        f"{str(r_pooled['p']):>10} {str(r_pooled['probe']):>10}"
    )

    # ─── 3. Sub-call motif structure ───
    all_sub = (
        np.concatenate([np.asarray(g[1], dtype=np.float32) for g in call_sub_groups])
        if call_sub_groups
        else np.zeros((0, fp_dim), dtype=np.float32)
    )
    if len(all_sub) >= args.k * 5:
        sub_parent = np.asarray([g[0] for g in call_sub_groups for _ in g[1]])
        r_sub = recovery(all_sub, sub_parent, args.k)
        print(
            f"{'sub-call (parent lbl)':24} {r_sub['V']:10.3f} {r_sub['nmi']:8.3f} "
            f"{str(r_sub['p']):>10} {str(r_sub['probe']):>10}"
        )

        # Per-call sub-call token sequences → LZ (within-call structural richness)
        scaler, cents = kmeans_codebook(all_sub, args.k)
        lz_vals = []
        for _lab, sfps in call_sub_groups:
            if len(sfps) < 2:
                continue
            toks = tokenize(np.asarray(sfps, dtype=np.float32), scaler, cents)
            lz_vals.append(lempel_ziv_complexity([int(t) for t in toks]).normalized)
        if lz_vals:
            print(
                f"\n  sub-call structure: {len(all_sub)} sub-segs across "
                f"{n_calls_with_internal} calls"
            )
            print(
                f"  per-call sub-call LZ76 (normalized): mean={np.mean(lz_vals):.3f} "
                f"median={np.median(lz_vals):.3f} (within-call token richness)"
            )
            print(
                f"  interpretation: sub-call V={r_sub['V']:.3f} vs whole-call V={r_whole['V']:.3f} "
                f"→ internal sub-call classes are "
                f"{'LESS' if r_sub['V'] < r_whole['V'] else 'MORE'} type-discriminative"
            )
    else:
        print(f"\n  (too few sub-segs ({len(all_sub)}) for sub-call recovery test)")

    print("HECB DOWNSTREAM DONE")


if __name__ == "__main__":
    main()
