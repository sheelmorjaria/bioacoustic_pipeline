#!/usr/bin/env python3
"""Zebra-finch external natural-unit benchmark — the frozen procedure's
portability test on a second independently annotated bird corpus.
Advisor-directed order (2026-10-20): Bengalese decomposition -> freeze ->
THIS benchmark -> stop the unit-recovery branch.

DECISION TREE (preregistered verbatim from the directive):
  If natural-unit localization and same-class recovery are good, the
  Bengalese failure is corpus/procedure-interaction specific. If they
  are poor again, the current fingerprint+DTW discovery method is not
  validated as a natural-unit detector across two independently
  annotated bird corpora. In either case, no threshold or representation
  changes are made during the benchmark.
REFINED QUESTION (post-decomposition): does the same THREE-STAGE PATTERN
hold — good oracle-boundary retrieval, recoverable proposal recall, weak
edge typing? Each stage is reported separately; the branch then STOPS.

MATERIAL: Lee et al. 2026 benchmark data (Zenodo 18156254, CC-BY-4.0,
md5-verified at download): 3 juvenile male zebra finches (R3406/R3640/
R3822), WAVs (44.1 kHz) + THREE independent expert manual clusterings
(manual_expr_{az,kl,mb}). REFERENCE = expert 'az' (first alphabetically
— prespecified); 'kl'/'mb' used only for inter-expert agreement QA.
PROVENANCE (declared): segment boundaries are WhisperSeg-proposed with
expert CLUSTER labels — the class reference is expert, the boundary
provenance is machine. Scope note: juvenile plastic song.

FROZEN PROCEDURE (identical machinery to the Bengalese studies; no
threshold or representation changes): 12-feature fingerprint contract
(40 ms/10 ms, species zebra_finch), per-bird discovery-stratum scaling,
NBD regimes A1_raw (0.05/0.03, drop off, 30 ms) and A2_highrecall
(0.02/0.015, drop 0.015, 20 ms), 4-way typed selector (onset/offset/
within/gap; kept = onset+offset) trained on DEVELOPMENT labels only
(logistic, balanced, 13 standardised proposal features).

SPLITS: per bird, sorted unique recording files, i%5 rule (0,1 ->
discovery, 2 -> calibration, else validation) — the programme's frozen
convention. Labels hidden during retrieval/proposal stages; used only
for typing-selector training (dev) and scoring.

ENDPOINTS (fixed now):
  E1 R_oracle:    GT segments (discovery -> validation, cross-file),
                  fixed-endpoint DTW ranking: Recall@1/@10 same-class,
                  majority-class chance, segment-clustered bootstrap.
  E2 R_proposal:  boundary recall at tol 3 frames per NBD regime,
                  proposals/sec, gap fraction (does the high-recall
                  regime transfer, or is it corpus-specific as locked in).
  E3 typing:      kept P/R/F1 + edge purity delta (raw -> selected) on
                  validation; typed vs the keep-everything reference.
  QA inter-expert agreement on shared segments (az vs kl/mb).
PATTERN VERDICTS: oracle_good (R@1 >= 0.5); proposal_recoverable
(best-regime recall >= 0.8); typing_weak (kept F1 < 0.6).

Usage:
    python scripts/zebrafinch_natural_unit_benchmark.py --self-test
    python scripts/zebrafinch_natural_unit_benchmark.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bengalese_boundary_decomposition import TOL_PRIMARY, _match_boundaries
from bengalese_proposal_selection import FEATURES, REGIMES, _label_proposals
from bengalese_score_diagnostics import dtw_full
from natural_unit_reference_benchmark import _boot_rate

DATA = ROOT / "data" / "zf_clustering"
BIRDS = ("R3406", "R3640", "R3822")
REF_EXPERT = "az"
SR = 44100
WIN_MS, HOP_MS = 40.0, 10.0
SPECIES = "zebra_finch"
MIN_FRAMES = 3
BOOT_B = 1000
SEED = 0


def _load_bird(bird):
    """az-reference segments + wav paths; None if h5py/wavs unavailable."""
    import h5py

    bdir = DATA / bird
    h5p = bdir / f"{bird}_manual_expr_{REF_EXPERT}.h5"
    if not h5p.exists():
        return None
    segs_by_file: dict = {}
    with h5py.File(h5p, "r") as f:
        fnames = [x.decode() if isinstance(x, bytes) else str(x) for x in f["files"]["filename"][:]]
        cid = f["segments"]["cluster_id"][:]
        onset = f["segments"]["onset_sec"][:]
        dur = f["segments"]["duration_sec"][:]
        fid = f["segments"]["file_id"][:]
    n_out_of_range = 0
    for c, o, d, j in zip(cid, onset, dur, fid):
        j = int(j)
        if not 0 <= j < len(fnames):
            n_out_of_range += 1  # segment references a file outside the table — skipped, reported
            continue
        segs_by_file.setdefault(fnames[j], []).append(
            {"label": f"c{int(c)}", "t0": float(o), "t1": float(o) + float(d)}
        )
    files = {}
    for w in sorted((bdir / "wavs").glob("*.wav")):
        files[w.name] = w
    return {"segs": segs_by_file, "wavs": files, "n_file_id_out_of_range": n_out_of_range}


def _expert_agreement(bird):
    """QA: label agreement between the reference expert and the others on
    shared segments (matched by file + onset within 5 ms)."""
    import h5py

    out = {}
    for other in ("kl", "mb"):
        p = DATA / bird / f"{bird}_manual_expr_{other}.h5"
        if not p.exists():
            continue
        with (
            h5py.File(p, "r") as f1,
            h5py.File(DATA / bird / f"{bird}_manual_expr_{REF_EXPERT}.h5") as f2,
        ):
            fn = [
                x.decode() if isinstance(x, bytes) else str(x) for x in f1["files"]["filename"][:]
            ]
            segs1 = {}
            for c, o, fid in zip(
                f1["segments"]["cluster_id"][:],
                f1["segments"]["onset_sec"][:],
                f1["segments"]["file_id"][:],
            ):
                if 0 <= int(fid) < len(fn):
                    segs1.setdefault(fn[int(fid)], []).append((round(float(o), 3), int(c)))
            fn2 = [
                x.decode() if isinstance(x, bytes) else str(x) for x in f2["files"]["filename"][:]
            ]
            segs2 = {}
            for c, o, fid in zip(
                f2["segments"]["cluster_id"][:],
                f2["segments"]["onset_sec"][:],
                f2["segments"]["file_id"][:],
            ):
                if 0 <= int(fid) < len(fn2):
                    segs2.setdefault(fn2[int(fid)], []).append((round(float(o), 3), int(c)))
        n_shared = n_agree = 0
        for fname, lst in segs1.items():
            for o, c in lst:
                match = next(
                    ((o2, c2) for o2, c2 in segs2.get(fname, []) if abs(o2 - o) < 0.005), None
                )
                if match:
                    n_shared += 1
                    n_agree += int(match[1] == c)
        out[f"az_vs_{other}"] = {
            "n_shared_segments": n_shared,
            "label_agreement": round(n_agree / max(n_shared, 1), 4),
        }
    return out


def _fingerprint_wav(args):
    """12-feature trajectory + valid mask for one wav (module-level)."""
    import numpy as np
    import soundfile as sf

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )

    name, path, hop, win = args
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    fidx = [_NAME_TO_IDX[n] for n in supported_features()]
    rows, i = [], 0
    while i * hop + win <= len(audio) and len(rows) < 4000:
        v = afp.extract(audio[i * hop : i * hop + win], sr, species=SPECIES).combined
        rows.append(np.asarray(v, dtype=np.float64)[fidx])
        i += 1
    if len(rows) < MIN_FRAMES:
        return name, None, None
    tr = np.stack(rows)
    return name, tr, ~np.all(tr == 0, axis=1)


def _ann_frames(segs, hop, win, n_frames):
    """Expert segment spans -> supported frame intervals (center rule)."""
    import numpy as np

    centers = np.arange(n_frames) * hop + win / 2.0
    out, n_unsup = [], 0
    for s in segs:
        idx = np.where((centers >= s["t0"] * SR) & (centers < s["t1"] * SR))[0]
        if len(idx) < MIN_FRAMES:
            n_unsup += 1
            continue
        out.append({"label": s["label"], "s": int(idx[0]), "e": int(idx[-1]) + 1})
    return out, n_unsup


def _proposals_wav(path, regime):
    import numpy as np
    import soundfile as sf

    from bioacoustic_pipeline.segmentation import nbd_proposal as nbd

    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sig = nbd.compute_signals(audio, sr, hop_ms=HOP_MS)
    props = nbd.generate_proposals(sig, **REGIMES[regime])
    X = [
        [
            p.energy_change,
            p.spectral_flux,
            p.band_flux,
            np.log1p(p.local_rms),
            p.rms_slope,
            p.duration_since_prev_ms,
            p.rms_ratio,
            p.flux_contrast,
            p.energy_contrast,
            p.flux_asym,
            p.duration_to_next_ms,
            p.spectral_drop,
            float("energy" in p.detector),
            float("spectral" in p.detector),
            float("spectral_drop" in p.detector),
        ]
        for p in props
    ]
    return {
        "X": np.array(X) if X else np.zeros((0, len(FEATURES))),
        "frames": [p.frame for p in props],
        "n_frames": sig.n_frames,
        "duration_s": len(audio) / sr,
    }


def _run_bird(bird):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    d = _load_bird(bird)
    if d is None:
        return None
    hop, win = int(SR * HOP_MS / 1000), int(SR * WIN_MS / 1000)
    names = sorted(n for n in d["segs"] if n in d["wavs"])
    if len(names) < 10:
        return None
    with ProcessPoolExecutor(max_workers=24) as ex:
        tracks = {
            n: (t, v)
            for n, t, v in ex.map(
                _fingerprint_wav, [(n, d["wavs"][n], hop, win) for n in names], chunksize=1
            )
            if t is not None
        }
    files = []
    for i, n in enumerate(names):
        if n not in tracks:
            continue
        tr, valid = tracks[n]
        ann, n_unsup = _ann_frames(d["segs"][n], hop, win, len(tr))
        files.append(
            {
                "name": n,
                "stage": "discovery"
                if i % 5 in (0, 1)
                else "calibration"
                if i % 5 == 2
                else "validation",
                "track": tr,
                "valid": valid,
                "ann": ann,
                "n_unsupported": n_unsup,
                "wav": str(d["wavs"][n]),
            }
        )
    parts = {
        s: [f for f in files if f["stage"] == s] for s in ("discovery", "calibration", "validation")
    }
    if min(map(len, parts.values())) < 2:
        return None
    disc = np.vstack([f["track"] for f in parts["discovery"]])
    mu, sd = disc.mean(0), disc.std(0) + 1e-9
    for f in files:
        f["z"] = (f["track"] - mu) / sd

    # ---- E1 oracle-boundary same-class retrieval (cross-file)
    def _segs(fs):
        out = []
        for f in fs:
            for a in f["ann"]:
                seg = f["z"][a["s"] : a["e"]]
                if len(seg) >= MIN_FRAMES:
                    out.append({"label": a["label"], "file": f["name"], "z": seg})
        return out

    q_segs, v_segs = _segs(parts["discovery"]), _segs(parts["validation"])
    per_tok = []
    for q in q_segs:
        costs = []
        for j, v in enumerate(v_segs):
            if v["file"] == q["file"]:
                continue
            c = dtw_full(q["z"], v["z"])
            if np.isfinite(c):
                costs.append((c, j))
        if not costs:
            continue
        costs.sort(key=lambda r: r[0])
        per_tok.append(
            (
                1 if v_segs[costs[0][1]]["label"] == q["label"] else 0,
                1 if any(v_segs[j]["label"] == q["label"] for _, j in costs[:10]) else 0,
            )
        )
    prev = {}
    for v in v_segs:
        prev[v["label"]] = prev.get(v["label"], 0) + 1
    e1 = {
        "n_query": len(per_tok),
        "n_validation_segments": len(v_segs),
        "n_classes": len(prev),
        "recall1": _boot_rate([(a, 1) for a, _ in per_tok], b=BOOT_B) if per_tok else None,
        "recall10": _boot_rate([(b, 1) for _, b in per_tok], b=BOOT_B) if per_tok else None,
        "majority_class_chance": round(max(prev.values()) / len(v_segs), 4) if v_segs else None,
    }

    # ---- E2 proposal recall per regime (labels only for scoring)
    prop = {}
    for regime in ("A1_raw", "A2_highrecall"):
        n_m = n_true = n_prop = n_gap = 0
        dur = 0.0
        for f in files:
            pr = _proposals_wav(f["wav"], regime)
            f[regime] = pr
            dur += pr["duration_s"]
            true_union = sorted({a["s"] for a in f["ann"]} | {a["e"] - 1 for a in f["ann"]})
            n_true += len(true_union)
            n_prop += len(pr["frames"])
            pairs, _, _ = _match_boundaries(true_union, pr["frames"], TOL_PRIMARY)
            n_m += len(pairs)
            n_gap += sum(
                1 for fr in pr["frames"] if not any(a["s"] <= fr < a["e"] for a in f["ann"])
            )
        prop[regime] = {
            "recall": round(n_m / max(n_true, 1), 4),
            "proposals_per_sec": round(n_prop / max(dur, 1e-9), 2),
            "gap_fraction": round(n_gap / max(n_prop, 1), 4),
        }

    # ---- E3 typed selector trained on DEVELOPMENT labels (A2 frozen)
    dev = parts["discovery"] + parts["calibration"]
    Xd, yd = [], []
    for f in dev:
        pr = f["A2_highrecall"]
        Xd.append(pr["X"])
        yd.extend(_label_proposals(pr["frames"], f["ann"]))
    Xd = np.vstack(Xd)
    scaler = StandardScaler().fit(Xd)
    typed = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(
        scaler.transform(Xd), yd
    )
    n_kept = n_true = n_m = sel_edge = sel_n = 0
    for f in parts["validation"]:
        pr = f["A2_highrecall"]
        pred = typed.predict(scaler.transform(pr["X"]))
        kept = [fr for fr, p in zip(pr["frames"], pred) if p in ("onset", "offset")]
        n_kept += len(kept)
        true_union = sorted({a["s"] for a in f["ann"]} | {a["e"] - 1 for a in f["ann"]})
        n_true += len(true_union)
        pairs, _, _ = _match_boundaries(true_union, kept, TOL_PRIMARY)
        n_m += len(pairs)
        labs = _label_proposals(kept, f["ann"])
        sel_edge += sum(x in ("onset", "offset") for x in labs)
        sel_n += len(kept)
    P = n_m / max(n_kept, 1)
    R = n_m / max(n_true, 1)
    a2_gap = prop["A2_highrecall"]["gap_fraction"]
    raw_purity_proxy = 1 - a2_gap
    e3 = {
        "kept_precision": round(P, 4),
        "kept_recall": round(R, 4),
        "kept_f1": round(2 * P * R / max(P + R, 1e-9), 4),
        "selected_edge_purity": round(sel_edge / max(sel_n, 1), 4) if sel_n else None,
        "raw_edge_purity_proxy": round(raw_purity_proxy, 4),
    }

    return {
        "n_files": len(files),
        "stages": {k: len(v) for k, v in parts.items()},
        "n_file_id_out_of_range": d.get("n_file_id_out_of_range", 0),
        "n_unsupported_segments": int(sum(f["n_unsupported"] for f in files)),
        "e1_oracle_retrieval": e1,
        "e2_proposal": prop,
        "e3_typing": e3,
        "expert_agreement_qa": _expert_agreement(bird),
        "pattern_verdicts": {
            "oracle_good": bool(e1["recall1"] and e1["recall1"]["point"] >= 0.5),
            "proposal_recoverable": bool(max(prop[r]["recall"] for r in prop) >= 0.8),
            "typing_weak": bool(e3["kept_f1"] < 0.6),
        },
    }


def _self_test() -> None:
    """Mechanics: split rule, ann mapping, oracle retrieval separation."""
    import numpy as np

    segs = [{"label": "a", "t0": 0.10, "t1": 0.22}, {"label": "b", "t0": 0.40, "t1": 0.55}]
    ann, n_unsup = _ann_frames(segs, 441, 1764, 100)
    assert len(ann) == 2 and ann[0]["label"] == "a" and ann[0]["s"] < ann[0]["e"], ann
    segA = np.stack([np.sin(np.linspace(0, 6 * np.pi, 12))] * 12, axis=1)
    segB = np.stack([np.cos(np.linspace(0, 2 * np.pi, 12))] * 12, axis=1)
    rng = np.random.RandomState(0)
    assert dtw_full(segA, segA + 0.01 * rng.randn(*segA.shape)) < dtw_full(segA, segB)
    print(
        f"SELF-TEST OK (ann mapping {[(a['label'], a['s'], a['e']) for a in ann]}, "
        "oracle retrieval separates classes)",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return

    per_bird = {}
    for bird in BIRDS:
        r = _run_bird(bird)
        if r is None:
            print(f"[{bird}] skipped (missing data / insufficient split)", flush=True)
            continue
        per_bird[bird] = r
        e1, e2, e3 = r["e1_oracle_retrieval"], r["e2_proposal"], r["e3_typing"]
        print(
            f"[{bird}] files {r['n_files']} ({r['stages']}) | "
            f"E1 R@1 {e1['recall1']['point'] if e1['recall1'] else None} (chance {e1['majority_class_chance']}) | "
            f"E2 A1 {e2['A1_raw']['recall']} A2 {e2['A2_highrecall']['recall']} | "
            f"E3 F1 {e3['kept_f1']}",
            flush=True,
        )

    report = {
        "study": "zebra-finch external natural-unit benchmark (frozen "
        "procedure portability; unit-recovery branch STOPS after this "
        "per the directive)",
        "decision_tree_preregistered": "good localisation+recovery -> "
        "Bengalese failure corpus/procedure-interaction specific; poor "
        "again -> fingerprint+DTW not validated across two annotated "
        "bird corpora; no threshold/representation changes during the "
        "benchmark",
        "refined_question": "does the three-stage pattern hold — good "
        "oracle retrieval, recoverable proposal recall, weak edge typing",
        "provenance": "Lee et al. 2026 (Zenodo 18156254, CC-BY-4.0, md5 "
        "verified): 3 JUVENILE males; WhisperSeg-proposed boundaries with "
        "expert cluster labels (reference = expert az, prespecified; "
        "kl/mb = agreement QA); scope note: plastic song",
        "per_bird": per_bird,
        "stop_rule": "one run; branch frozen after",
    }
    out = ROOT / "outputs" / "zebrafinch_natural_unit_benchmark.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({b: r["pattern_verdicts"] for b, r in per_bird.items()}, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
