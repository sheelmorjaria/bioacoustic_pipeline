#!/usr/bin/env python3
"""Boundary-quality sensitivity sweep (Architecture v1.0, Step 2) — two corpora.

Paper question: **at what parent (call-event) boundary quality does sub-call
internal annotation help the downstream representation?** Run on TWO corpora with
complementary structure so neither a firewall lock nor a sparsity false-negative
dominates the conclusion:

  * **MeerKAT** (the HECB corpus): brief calls in noise + a trained CRNN, so parent
    quality varies via REAL detector operating points (CRNN enter_thr) and the typed
    HECB internal annotator is available. Methods: oracle/CRNN(0.4/0.7/0.9)/raw-NBD
    parent × typed-HECB/raw-NBD internal. (HECB typed is firewall-locked to meerkat.)
  * **Marmoset** (richer ~1 s calls, 5 call types): pre-extracted single calls ⇒ the
    parent is perfect by construction, so parent quality is degraded by GAUSSIAN
    onset/offset jitter (σ = 0/25/75/150 ms). No HECB checkpoint ⇒ raw-NBD internal
    only. This is where a hierarchy benefit is most likely to show if MeerKAT's brief
    calls prove too atomic.

Shared scoring: build_vocalisation → whole_sub_pooled_sequence_ablation (whole/pooled/
sequence V + probe; + summarize_ablation winner/claim_strength) + DUSTED motif discovery
(family_entropy, top_family_fraction, bootstrap_stability) on the ≥2-sub subset.

Read: the hierarchy-helps signal is the GAP (pooled−whole V). As parent quality
degrades, the gap should shrink ⇒ the parent-quality threshold below which HECB should
abstain (``min_parent_confidence``). MeerKAT x-axis = parent onset F1; marmoset x-axis =
parent boundary MAE (ms).

Honest guardrails: ``n_with_internal`` reported per cell; DUSTED runs only on the ≥2-sub
subset. Call-type recovery is an ACOUSTIC association, NOT validated meaning
(``semantic_validation`` stays ``none``). All-species fingerprint + HECB/raw-NBD only
(firewall-clean).

Run:  python scripts/boundary_quality_sweep.py --n-clips 300 --n-per-type 2500

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import track_b_meerkat as MK  # noqa: E402

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.diagnostics.motif_miner import subsequence_motifs as sm  # noqa: E402
from bioacoustic_pipeline.ingestion.manifest import load_species_profiles  # noqa: E402
from bioacoustic_pipeline.representation.vocalisation import (  # noqa: E402
    build_vocalisation,
    summarize_ablation,
    whole_sub_pooled_sequence_ablation,
)
from bioacoustic_pipeline.segmentation.hierarchical import (  # noqa: E402
    CallEvent,
    HECBDurationPriors,
    hecb_decode,
    segment_call_internally,
)
from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_one_to_one  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

K_ABLATION = 12
K_SUB = 24
IOU_MIN = 0.3
EVAL_TOL_MS = 100.0
MIN_INTERNAL_MS = 250.0

# Marmoset corpus (mirror scripts/track_b_marmoset.py).
MAR_VOC_DIR = Path("/home/sheel/birdsong_analysis/data/Vocalizations")
MAR_ANNOT = MAR_VOC_DIR / "Annotations.tsv"
MAR_CALL_TYPES = ("Phee", "Twitter", "Trill", "Tsik", "Seep")
MAR_SR = 96000

# MeerKAT methods: (name, parent_source, internal_kind, enter_thr).
MK_METHODS = [
    ("oracle_typed", "oracle", "typed", None),
    ("oracle_raw", "oracle", "raw", None),
    ("crnn0.7_typed", "crnn", "typed", 0.7),
    ("crnn0.7_raw", "crnn", "raw", 0.7),
    ("crnn0.9_typed", "crnn", "typed", 0.9),
    ("crnn0.4_typed", "crnn", "typed", 0.4),
    ("raw_nbd", "nbd", "none", None),
]
# Marmoset methods: (name, internal_kind, jitter_sigma_ms).
MAR_METHODS = [
    ("oracle_raw", "raw", 0.0),
    ("jitter25_raw", "raw", 25.0),
    ("jitter75_raw", "raw", 75.0),
    ("jitter150_raw", "raw", 150.0),
    ("oracle_none", "none", 0.0),
]

# Late-bound HECB private helpers (bound in main after the torch-availability check).
_proba_to_events = None
_event_conditioned_proposals = None


def _bind_hecb_helpers(hecb_mod):
    global _proba_to_events, _event_conditioned_proposals  # noqa: PLW0603
    _crnn_proba = hecb_mod._crnn_proba
    _proba_to_events = hecb_mod._proba_to_events
    _event_conditioned_proposals = hecb_mod._event_conditioned_proposals
    return _crnn_proba


# ─── shared scoring helpers ─────────────────────────────────────────────────


def iou(a_on, a_off, b_on, b_off):
    inter = max(0.0, min(a_off, b_off) - max(a_on, b_on))
    union = max(a_off, b_off) - min(a_on, b_on)
    return inter / union if union > 0 else 0.0


def match_to_gt(calls, gt_calls):
    """Greedy max-IoU match CallEvents → GT (onset_s, offset_s, label); each GT once."""
    available = list(range(len(gt_calls)))
    matched = []
    for ce in calls:
        best_j, best_iou = -1, 0.0
        for j in available:
            g_on, g_off, _ = gt_calls[j]
            v = iou(ce.onset_s, ce.offset_s, g_on, g_off)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= IOU_MIN:
            matched.append((ce, gt_calls[best_j][2]))
            available.remove(best_j)
    return matched


def onset_f1(events_ms, gt_calls):
    pred = [on for on, _off, _c in events_ms]
    gt = [s * 1000.0 for s, _e, _l in gt_calls]
    if not pred or not gt:
        return 0.0
    r = prf_one_to_one(pred, gt, tol=EVAL_TOL_MS)
    p = r.tp / max(r.tp + r.fp, 1)
    rec = r.tp / max(r.tp + r.fn, 1)
    return 2 * p * rec / max(p + rec, 1e-8)


def fit_subcall_tokenizer(vocs, k):
    all_sub = [
        np.asarray(s.acoustic_fingerprint, dtype=np.float32)
        for v in vocs
        for s in v.sub_segments
        if s.acoustic_fingerprint.size
    ]
    if len(all_sub) < k * 5:
        return None
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    X = np.asarray(all_sub, dtype=np.float32)
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(scaler.transform(X))
    return scaler, km.cluster_centers_


def _tokenize(vecs, scaler, centers):
    sv = scaler.transform(np.asarray(vecs, dtype=np.float32))
    return np.argmin(((sv[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)


def run_dusted(vocs, k_sub, n_motif_sample, seed):
    """DUSTED on the ≥2-sub-call subset → summary dict (+bootstrap stability)."""
    sc = fit_subcall_tokenizer(vocs, k_sub)
    if sc is None:
        return _empty_dusted(0)
    scaler, centers = sc
    calls = []
    for i, v in enumerate(vocs):
        sfps = [
            np.asarray(s.acoustic_fingerprint, dtype=np.float32)
            for s in v.sub_segments
            if s.acoustic_fingerprint.size
        ]
        if len(sfps) >= 2:
            calls.append(
                sm.CallUnits(
                    fingerprints=sfps,
                    clusters=list(_tokenize(sfps, scaler, centers)),
                    calltype=v.calltype,
                    file_id=v.vocalisation_id,
                    call_idx=i,
                )
            )
    if len(calls) < 30:
        return _empty_dusted(len(calls))
    sample = calls[:n_motif_sample] if len(calls) > n_motif_sample else calls
    common = dict(min_n=2, max_n=4, min_support=3, min_calls=2)
    motifs = sm.discover_recurrent_subsequences(
        sample, **common, dist_threshold=0.15, centroid_radius=0.20
    )
    s = sm.summarize_discovery(motifs, len(sample))
    try:
        stab = sm.bootstrap_stability(
            sample,
            dist_threshold=0.15,
            centroid_radius=0.20,
            n_boot=20,
            top_k=10,
            seed=seed,
            **common,
        )
    except Exception:  # noqa: BLE001
        stab = None
    return {
        "n_dusted_calls": len(calls),
        "n_sampled": len(sample),
        "dropped": len(calls) - len(sample),
        "n_motifs": s["n_motifs"],
        "family_entropy": s.get("family_entropy_norm", s.get("family_entropy")),
        "top_family_fraction": s.get("top_family_support_fraction"),
        "coverage": s.get("coverage"),
        "bootstrap_stability": stab,
    }


def _empty_dusted(n):
    return {
        "n_dusted_calls": n,
        "n_sampled": None,
        "dropped": None,
        "n_motifs": None,
        "family_entropy": None,
        "top_family_fraction": None,
        "coverage": None,
        "bootstrap_stability": None,
    }


def _probe(rec):
    return None if rec is None or rec.get("probe") is None else round(rec["probe"], 3)


def score_method(corpus, method, vocs, labels, parent_quality, args):
    """Shared per-method scoring → one row dict."""
    n_with_internal = sum(1 for v in vocs if v.subcall_count >= 2)
    row = {
        "corpus": corpus,
        "method": method,
        "n_calls": len(vocs),
        "n_with_internal": n_with_internal,
        "parent_f1": parent_quality.get("f1"),
        "parent_mae_ms": parent_quality.get("mae_ms"),
    }
    if len(vocs) >= 10:
        res = whole_sub_pooled_sequence_ablation(vocs, labels, k=K_ABLATION)
        ab = summarize_ablation(res, metric="V")
        row.update(
            {
                "whole_V": round(res["whole"]["V"], 3),
                "pooled_V": round(res["pooled"]["V"], 3),
                "sequence_V": round(res["sequence"]["V"], 3),
                "whole_probe": _probe(res["whole"]),
                "pooled_probe": _probe(res["pooled"]),
                "sequence_probe": _probe(res["sequence"]),
                "pooled_minus_whole_V": round(res["pooled"]["V"] - res["whole"]["V"], 3),
                "winner": ab.winner,
                "claim_strength": ab.claim_strength,
            }
        )
        dust = run_dusted(vocs, K_SUB, args.n_motif_sample, args.seed)
        row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in dust.items()})
    else:
        for k in (
            "whole_V",
            "pooled_V",
            "sequence_V",
            "whole_probe",
            "pooled_probe",
            "sequence_probe",
            "pooled_minus_whole_V",
            "winner",
            "claim_strength",
            "n_dusted_calls",
            "n_sampled",
            "dropped",
            "n_motifs",
            "family_entropy",
            "top_family_fraction",
            "coverage",
            "bootstrap_stability",
        ):
            row[k] = None
    return row


# ─── MeerKAT runner ─────────────────────────────────────────────────────────


def _mk_events(method, audio, sr, proba, gt_calls, clf, dsp, profile):
    _name, parent_src, internal_kind, enter_thr = method
    if parent_src == "oracle":
        events = [(s * 1000.0, e * 1000.0, 1.0) for s, e, _l in gt_calls]
    elif parent_src == "crnn":
        dsp_thr = replace(dsp, enter=enter_thr, exit_=min(enter_thr * 0.4, enter_thr - 0.05))
        events = _proba_to_events(proba, dsp_thr)
    else:  # nbd
        from bioacoustic_pipeline.orchestration import segment_audio  # noqa: PLC0415

        segs = segment_audio(audio, sr, profile, calibrator=None)
        events = [(s.start_ms, s.end_ms, 1.0) for s in segs if s.end_ms > s.start_ms]
    if internal_kind == "typed":
        props = _event_conditioned_proposals(audio, sr, proba, events, profile, dsp, None)
        typed = [(t, clf.predict_proba(ev.reshape(1, -1))[0]) for t, ev in props]
        calls = hecb_decode(events, typed, HECBDurationPriors(), snap=False, file_id=parent_src)
    else:  # raw or none
        calls = []
        for i, (on, off, conf) in enumerate(events):
            ce = CallEvent(
                call_id=f"{parent_src}{i}",
                onset_s=on / 1000.0,
                offset_s=off / 1000.0,
                crnn_confidence=float(conf),
            )
            if internal_kind == "raw":
                ce.internal_segments = segment_call_internally(
                    ce, audio, sr, profile, min_call_duration_ms=MIN_INTERNAL_MS
                )
            calls.append(ce)
    return events, calls


def run_meerkat(args, hecb_mod, _crnn_proba):
    crnn, clf, dsp = hecb_mod.load_hecb("meerkat", "cpu")
    if crnn is None:
        print("[meerkat] no HECB checkpoint — skipping", flush=True)
        return []
    profile = load_species_profiles()["meerkat"]
    fp_fn = lambda seg, sr: np.asarray(  # noqa: E731
        afp.extract(seg, sr, species="meerkat").combined, dtype=np.float32
    )
    clips = MK.sample_clips(args.n_clips, args.seed)
    print(f"[meerkat] {len(clips)} clips", flush=True)

    accum = {m[0]: {"vocs": [], "labels": [], "events": []} for m in MK_METHODS}
    all_gt = []
    for ci, (h5path, gt_calls) in enumerate(clips):
        wav = MK.WAV_DIR / (Path(h5path).stem + ".wav")
        try:
            audio, sr = load_audio(str(wav), target_sr=profile.get("sample_rate"))
        except Exception:  # noqa: BLE001
            continue
        if audio is None or audio.size == 0:
            continue
        all_gt.extend(gt_calls)
        proba = _crnn_proba(crnn, audio, dsp, "cpu")
        for m in MK_METHODS:
            events, calls = _mk_events(m, audio, sr, proba, gt_calls, clf, dsp, profile)
            accum[m[0]]["events"].extend(events)
            for ce, label in match_to_gt(calls, gt_calls):
                voc = build_vocalisation(
                    ce,
                    audio,
                    sr,
                    fingerprint_fn=fp_fn,
                    species="meerkat",
                    calltype=label,
                    file_id=Path(h5path).stem,
                )
                if voc is not None:
                    accum[m[0]]["vocs"].append(voc)
                    accum[m[0]]["labels"].append(label)
        if (ci + 1) % 25 == 0:
            print(f"  [meerkat] {ci + 1}/{len(clips)} clips", flush=True)

    rows = []
    for m in MK_METHODS:
        a = accum[m[0]]
        pq = {"f1": round(onset_f1(a["events"], all_gt), 3), "mae_ms": None}
        rows.append(score_method("meerkat", m[0], a["vocs"], a["labels"], pq, args))
    return rows


# ─── Marmoset runner ────────────────────────────────────────────────────────


def _mar_load_index():
    idx = {}
    with MAR_ANNOT.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            fn = (row.get("file_name") or "").strip()
            if fn:
                parent = (row.get("parent_name") or "").strip()
                idx[fn] = (parent, row.get("label", "").strip())
    return idx


def sample_marmoset(n_per_type, seed):
    idx = _mar_load_index()
    by_type = defaultdict(list)
    for fn, (parent, label) in idx.items():
        if label in MAR_CALL_TYPES:
            by_type[label].append((parent, fn))
    rng = random.Random(seed)
    out = []
    for ct in MAR_CALL_TYPES:
        pool = list(by_type.get(ct, []))
        rng.shuffle(pool)
        for parent, fn in pool[:n_per_type]:
            path = MAR_VOC_DIR / parent / fn
            if not path.exists():
                continue
            try:
                audio, sr = load_audio(str(path), target_sr=MAR_SR)
            except Exception:  # noqa: BLE001
                continue
            if audio is None or len(audio) < int(0.15 * MAR_SR):
                continue
            out.append((audio, sr, ct))
    return out


def run_marmoset(args):
    profile = load_species_profiles().get("marmoset")
    if not profile:
        print("[marmoset] no species profile — skipping", flush=True)
        return []
    fp_fn = lambda seg, sr: np.asarray(  # noqa: E731
        afp.extract(seg, sr, species="marmoset").combined, dtype=np.float32
    )
    calls = sample_marmoset(args.n_per_type, args.seed)
    print(
        f"[marmoset] {len(calls)} calls "
        f"({dict((c, sum(1 for _, _, x in calls if x == c)) for c in MAR_CALL_TYPES)})",
        flush=True,
    )

    rows = []
    for name, internal_kind, sigma in MAR_METHODS:
        rng = random.Random(args.seed + 1)
        vocs, labels, errs = [], [], []
        for ci, (audio, sr, label) in enumerate(calls):
            dur_ms = len(audio) / sr * 1000.0
            j_on = rng.uniform(-sigma, sigma) if sigma > 0 else 0.0
            j_off = rng.uniform(-sigma, sigma) if sigma > 0 else 0.0
            on_ms, off_ms = max(0.0, j_on), min(dur_ms, dur_ms + j_off)
            if off_ms - on_ms < 50.0:
                continue
            ce = CallEvent(call_id=f"{name}{ci}", onset_s=on_ms / 1000.0, offset_s=off_ms / 1000.0)
            if internal_kind == "raw":
                ce.internal_segments = segment_call_internally(
                    ce, audio, sr, profile, min_call_duration_ms=MIN_INTERNAL_MS
                )
            voc = build_vocalisation(
                ce,
                audio,
                sr,
                fingerprint_fn=fp_fn,
                species="marmoset",
                calltype=label,
                file_id=name,
            )
            if voc is not None:
                vocs.append(voc)
                labels.append(label)
            errs.append(abs(on_ms) + abs(off_ms - dur_ms))
        mae = round(float(np.mean(errs)), 1) if errs else None
        rows.append(score_method("marmoset", name, vocs, labels, {"mae_ms": mae}, args))
        print(
            f"  [marmoset] {name}: n={len(vocs)} internal={sum(1 for v in vocs if v.subcall_count >= 2)} "
            f"mae={mae}ms",
            flush=True,
        )
    return rows


# ─── output ─────────────────────────────────────────────────────────────────


def _print_table(rows):
    cols = [
        "corpus",
        "method",
        "parent_f1",
        "parent_mae_ms",
        "n_calls",
        "n_with_internal",
        "whole_V",
        "pooled_V",
        "sequence_V",
        "pooled_minus_whole_V",
        "whole_probe",
        "pooled_probe",
        "family_entropy",
        "top_family_fraction",
        "bootstrap_stability",
    ]
    print(
        f"\n{'=' * 150}\nBOUNDARY-QUALITY SWEEP — meerkat (HECB, real detector variation) "
        f"+ marmoset (oracle/jittered parent, raw-NBD internal)\n{'=' * 150}"
    )
    print("  " + " ".join(f"{c[:15]:>15}" for c in cols))
    print("  " + "-" * 146)
    cur = None
    for r in rows:
        if r["corpus"] != cur:
            cur = r["corpus"]
            print(f"  ── {cur} " + "─" * 130)
        print("  " + " ".join(_cell(r.get(c)) for c in cols))


def _cell(v):
    if v is None:
        return f"{'n/a':>15}"
    if isinstance(v, float):
        return f"{v:>15.3f}"
    return f"{str(v)[:15]:>15}"


def _verdict(rows):
    for corpus, methods in (("meerkat", MK_METHODS), ("marmoset", MAR_METHODS)):
        sub = [r for r in rows if r["corpus"] == corpus]
        ref_name = "oracle_typed" if corpus == "meerkat" else "oracle_raw"
        ref = next((r for r in sub if r["method"] == ref_name), None)
        if not ref or ref.get("pooled_minus_whole_V") is None:
            print(f"\n  [{corpus}] no oracle hierarchy gap to reference.")
            continue
        ref_gap = ref["pooled_minus_whole_V"]
        xkey = "parent_f1" if corpus == "meerkat" else "parent_mae_ms"
        print(f"\n  [{corpus}] oracle ({ref_name}) pooled−whole V gap = {ref_gap:+.3f}")
        if ref_gap <= 0:
            print(
                "    ⇒ whole-call PRIMARY holds: pooled does not beat whole even at the best "
                "parent quality — no hierarchy benefit to collapse (no abstain threshold)."
            )
            ranked = sorted(
                [r for r in sub if r.get("pooled_minus_whole_V") is not None],
                key=lambda r: r[xkey] if r.get(xkey) is not None else 9e9,
            )
            for r in ranked:
                print(
                    f"    {r['method']:>16} {xkey}={r[xkey]} gap={r['pooled_minus_whole_V']:+.3f}"
                )
            continue
        ranked = sorted(
            [
                r
                for r in sub
                if r.get("pooled_minus_whole_V") is not None and r["method"] != ref_name
            ],
            key=lambda r: r[xkey] if r.get(xkey) is not None else 9e9,
        )
        for r in ranked:
            gap = r["pooled_minus_whole_V"]
            frac = gap / ref_gap if ref_gap else float("nan")
            flag = "  ← collapses (abstain)" if frac < 0.25 else ""
            print(
                f"    {r['method']:>16} {xkey}={r[xkey]} gap={gap:+.3f} "
                f"({frac:.0%} of oracle){flag}"
            )


def main():
    ap = argparse.ArgumentParser(
        description="Boundary-quality sensitivity sweep (meerkat+marmoset)"
    )
    ap.add_argument("--n-clips", type=int, default=300)
    ap.add_argument("--n-per-type", type=int, default=2500)
    ap.add_argument("--n-motif-sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bioacoustic_pipeline.segmentation import hecb as hecb_mod  # noqa: PLC0415

    _crnn_proba = None
    if hecb_mod.available:
        _crnn_proba = _bind_hecb_helpers(hecb_mod)
    else:
        print(
            f"torch/torchaudio unavailable — HECB meerkat leg skipped ({hecb_mod.load_error})",
            flush=True,
        )

    rows = []
    if _crnn_proba is not None:
        rows += run_meerkat(args, hecb_mod, _crnn_proba)
    rows += run_marmoset(args)

    _print_table(rows)
    _verdict(rows)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "boundary_quality_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "boundary_quality_sweep.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nsaved outputs/boundary_quality_sweep.{{csv,json}} ({len(rows)} rows)")
    print("\nBOUNDARY QUALITY SWEEP DONE")


if __name__ == "__main__":
    main()
