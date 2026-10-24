#!/usr/bin/env python3
"""Natural-repetition benchmark on annotated Bengalese finch song.

QUESTION (review-directed): can the frozen PROCEDURE recognise separate
NATURAL renditions of an annotated acoustic unit — not just planted copies?
Two tests, kept separate (matching/localisation vs discovery):

  TEST 1 — natural known-query recovery: query = ONE NATURAL annotated
  occurrence (never copied anywhere); search the bird's independent
  validation bouts. Endpoints = REFERENCE ACCURACY: precision (accepted
  intervals overlapping the same annotated class), recall over the class's
  supported validation occurrences, boundary error vs annotated on/offset.

  TEST 2 — blind fragment discovery: labels and boundaries hidden from the
  procedure (used only for scoring): within-bird seeding + subsequence-DTW
  families on discovery bouts. Endpoints: family purity (majority annotated
  label), class coverage, families-per-class (merge/split), boundary
  agreement.

  PRIMARY ENDPOINT IS REFERENCE ACCURACY, NOT FREQUENCY: a common real unit
  need not beat random fragments' recovery to be correctly identified.
  Same-bird random fragments are a SUPPLEMENTARY matchability reference.

PROCEDURE PORTABILITY (not parameter portability): the calibration
ALGORITHM is fixed (5th-pct random-fragment radius, discovery-fitted
scaling) and re-estimated per bird from its OWN discovery/calibration
recordings. The bat radius (1.0796) is NOT transplanted. Annotated units
shorter than the window contract are UNSUPPORTED: reported, kept in
denominators, never padded.

SPLITS: bout-level hash 40/20/40 discovery/calibration/validation per bird
(all events of one bout stay together; discovery/calibration excluded from
validation by release contract). Birds selected by annotation support
(metadata), not by scores. Reports per individual AND the distribution
across individuals.

FRAMING: recovering annotated syllables validates acoustic discovery at
that level in this regime. It does NOT establish intra-call words, meaning,
or that the bat negative reflects biological absence.

Usage:
    python scripts/bengalese_natural_benchmark.py --self-test
    python scripts/bengalese_natural_benchmark.py [--birds 0 1 2] [--bouts 90]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data" / "bengalifinch"
SR = 32000
WIN_MS, HOP_MS = 40.0, 10.0
SPECIES = "bengalese_finch"
MIN_FRAMES = 3  # window-contract support for an annotated syllable
LENGTHS = (8, 16, 24)
CTRL_PCT = 5.0
TOP_CLASSES = 6
SEED = 0
DISC_PCT, CAL_PCT = 40, 20


def parse_annotation(xml_path: Path):
    """xwaves Annotation.xml -> [(wav, [(abs_start_sample, length, label_str)])]."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    bouts = []
    for seq in tree.getroot().findall("Sequence"):
        wav = seq.findtext("WaveFileName") or ""
        bout_pos = int(seq.findtext("Position") or "0")
        notes = []
        for nt in seq.findall("Note"):
            rel = int(nt.findtext("Position") or "0")
            length = int(nt.findtext("Length") or "0")
            label = (nt.findtext("Label") or "").strip()
            if label and label != "*":  # '*' marks intro notes in this corpus
                notes.append((bout_pos + rel, length, label))
        if notes:
            bouts.append((wav, notes))
    return bouts


def _load_wav(path: Path):
    import numpy as np

    try:
        with wave.open(str(path), "rb") as wf:
            n, ch, width, sr = (
                wf.getnframes(),
                wf.getnchannels(),
                wf.getsampwidth(),
                wf.getframerate(),
            )
            if ch != 1 or width != 2 or sr != SR:
                return None
            raw = wf.readframes(n)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    except (wave.Error, OSError):
        return None


def _stage(name: str) -> str:
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 100
    return (
        "discovery" if h < DISC_PCT else ("calibration" if h < DISC_PCT + CAL_PCT else "validation")
    )


def _bout_track(args):
    """Fingerprint trajectory (supported 12 features) for one bout."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )

    wav_path, notes, hop, win = args
    audio = _load_wav(wav_path)
    if audio is None or len(audio) < win:
        return None
    fidx = [_NAME_TO_IDX[n] for n in supported_features()]
    rows, i = [], 0
    while i * hop + win <= len(audio) and len(rows) < 4000:
        seg = audio[i * hop : i * hop + win]
        v = afp.extract(seg, SR, species=SPECIES).combined
        rows.append(np.asarray(v, dtype=np.float64)[fidx])
        i += 1
    if len(rows) < MIN_FRAMES:
        return None
    track = np.stack(rows)
    valid = ~np.all(track == 0, axis=1)
    # annotated syllables -> frame intervals (window center inside syllable)
    ann = []
    n_unsupported = 0
    for abs_start, length, label in notes:
        a = abs_start
        b = abs_start + length
        centers = np.arange(len(track)) * hop + win / 2.0
        m = (centers >= a) & (centers < b)
        idx = np.where(m)[0]
        if len(idx) < MIN_FRAMES:
            n_unsupported += 1  # reported, kept in the denominator
            continue
        ann.append({"label": label, "s": int(idx[0]), "e": int(idx[-1]) + 1})
    return {"track": track, "valid": valid, "ann": ann, "n_unsupported": n_unsupported}


def _bout_job(r):
    """Module-level wrapper (picklable): one bout + its wav name."""
    wav_path, notes, hop, win, w = r
    out = _bout_track((wav_path, notes, hop, win))
    return None if out is None else {**out, "wav": w}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    _real_run(args)


def _self_test() -> None:
    """Synthetic annotated song: two syllable classes recurring in noise."""
    import numpy as np

    import subcall_term_discovery as std

    rng = np.random.RandomState(0)
    sylA = np.sin(np.linspace(0, 8 * np.pi, 14))[:, None] * np.ones((1, 12))
    sylB = np.cos(np.linspace(0, 3 * np.pi, 14))[:, None] * np.ones((1, 12)) * 0.6
    bouts = []
    for k in range(8):
        parts, ann = [], []
        pos = 0
        for j in range(rng.randint(4, 7)):
            which = "A" if rng.rand() < 0.5 else "B"
            syl = (sylA if which == "A" else sylB) + 0.02 * rng.randn(14, 12)
            parts.append(rng.randn(6, 12) * 0.8)  # unannotated gap
            parts.append(syl)
            ann.append({"label": which, "s": pos + 6, "e": pos + 20})
            pos += 20
        track = np.vstack(parts)
        bouts.append(
            {"track": track, "valid": np.ones(len(track), bool), "ann": ann, "n_unsupported": 0}
        )
    allf = np.vstack([b["track"] for b in bouts])
    mu, sd = allf.mean(0), allf.std(0) + 1e-9
    for b in bouts:
        b["z"] = (b["track"] - mu) / sd
    qlab = bouts[0]["ann"][0]["label"]
    q = bouts[0]["z"][bouts[0]["ann"][0]["s"] : bouts[0]["ann"][0]["e"]]
    hits = []
    for b in bouts[1:]:
        c, s, e = std.subseq_dtw(q, b["z"])
        if c < 1.0:
            lab = next((a for a in b["ann"] if not (e <= a["s"] or s >= a["e"])), None)
            hits.append(lab["label"] if lab else None)
    n_ok = sum(1 for h in hits if h == qlab)
    assert n_ok >= len(hits) * 0.7, (qlab, hits)
    print(
        f"SELF-TEST OK (natural-query precision {n_ok}/{len(hits)} for class "
        f"{qlab} on synthetic song)",
        flush=True,
    )


def _real_run(args) -> None:
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    import subcall_term_discovery as std
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )

    hop, win = int(SR * HOP_MS / 1000), int(SR * WIN_MS / 1000)
    rng = np.random.RandomState(SEED)
    per_bird = {}

    for bird in args.birds:
        bdir = DATA / f"Bird{bird}"
        xml = bdir / "Annotation.xml"
        if not xml.exists():
            print(f"[Bird{bird}] missing {xml}", flush=True)
            continue
        bouts_raw = parse_annotation(xml)[: args.bouts]
        jobs = [(bdir / "Wave" / w, notes, hop, win, w) for w, notes in bouts_raw]
        with ProcessPoolExecutor(max_workers=24) as ex:
            loaded = [r for r in ex.map(_bout_job, jobs, chunksize=2) if r]
        # one WAV can back several Sequence bouts: stage is per unique FILE
        # (all bouts of a file inherit it); wav carried with each result
        # strata assignment over sorted unique wavs: balanced 40/20/40 and
        # deterministic (v2 split rule — v1 hash split left Bird1 with an
        # empty calibration stratum; support-driven, no test scores seen)
        uniq_wavs = sorted({b["wav"] for b in loaded})
        strata = {}
        for i, w in enumerate(uniq_wavs):
            strata[w] = (
                "discovery" if i % 5 in (0, 1) else "calibration" if i % 5 == 2 else "validation"
            )
        for b in loaded:
            b["stage"] = strata[b["wav"]]
        seq_count: dict = {}
        for b in loaded:
            seq_count[b["wav"]] = seq_count.get(b["wav"], 0) + 1
            b["bout_id"] = f"{b['wav']}#{seq_count[b['wav']]}"
        parts = {
            s: [b for b in loaded if b["stage"] == s]
            for s in ("discovery", "calibration", "validation")
        }
        n_unsup = sum(b["n_unsupported"] for b in loaded)
        n_ann = sum(len(b["ann"]) for b in loaded) + n_unsup
        print(
            f"[Bird{bird}] bouts {len(loaded)} ("
            f"{len(parts['discovery'])}/{len(parts['calibration'])}/"
            f"{len(parts['validation'])}) | annotated syllables {n_ann} "
            f"(unsupported {n_unsup})",
            flush=True,
        )
        if min(len(parts["discovery"]), len(parts["calibration"]), len(parts["validation"])) < 3:
            print(f"[Bird{bird}] insufficient split support — skipped", flush=True)
            continue

        uniq_files = uniq_wavs
        contract = ScientificRunContract(
            run_name=f"bengalese_natural_benchmark_Bird{bird}",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            masks={
                "source_validity": MaskProvenance(
                    kind="source_validity",
                    produced_by="fingerprint absolute gate",
                    win_ms=WIN_MS,
                    hop_ms=HOP_MS,
                    amplitude_stage="native waveform",
                ),
                "detected_activity": MaskProvenance(
                    kind="detected_activity",
                    produced_by="annotated syllable spans",
                    win_ms=WIN_MS,
                    hop_ms=HOP_MS,
                    amplitude_stage="native waveform",
                ),
            },
            source_manifest=source_manifest_from(
                uniq_files, [strata[f] for f in uniq_files], uniq_files
            ),
            eval_split={"provenance": "bout-level hash 40/20/40 within bird", "unit": "song bout"},
        )
        enforcement = raise_for_science(contract)

        # procedure portability: bird-own scaling + radius
        disc_frames = np.vstack([b["track"] for b in parts["discovery"]])
        mu, sd = disc_frames.mean(0), disc_frames.std(0) + 1e-9
        for b in loaded:
            b["z"] = (b["track"] - mu) / sd
        cal = parts["calibration"]
        costs = []
        for i in range(40):
            p = cal[rng.randint(0, len(cal))]
            L = LENGTHS[i % len(LENGTHS)]
            if len(p["z"]) < L + 2:
                continue
            s0 = int(rng.randint(0, len(p["z"]) - L))
            for k2 in rng.randint(0, len(cal), size=6):
                p2 = cal[k2]
                if p2["wav"] == p["wav"]:
                    continue
                costs.append(std.subseq_dtw(p["z"][s0 : s0 + L], p2["z"])[0])
        costs = [c for c in costs if np.isfinite(c)]
        radius = float(np.percentile(costs, CTRL_PCT))
        print(
            f"[Bird{bird}] procedure-portable radius {radius:.4f} "
            f"(re-estimated; bat radius NOT transplanted)",
            flush=True,
        )

        # ---- TEST 1: natural known-query recovery
        class_counts: dict = {}
        for b in parts["discovery"]:
            for a in b["ann"]:
                class_counts[a["label"]] = class_counts.get(a["label"], 0) + 1
        top = sorted(class_counts, key=lambda k: -class_counts[k])[:TOP_CLASSES]
        t1 = []
        for lab in top:
            qb = next(b for b in parts["discovery"] for a in b["ann"] if a["label"] == lab)
            qa = next(a for a in qb["ann"] if a["label"] == lab)
            q = qb["z"][qa["s"] : qa["e"]]
            accepted, correct, bnd = 0, 0, []
            n_val_occ = 0
            for b in parts["validation"]:
                occ = [a for a in b["ann"] if a["label"] == lab]
                n_val_occ += len(occ)
                c, s, e = std.subseq_dtw(q, b["z"])
                if np.isfinite(c) and c <= radius:
                    accepted += 1
                    best = None
                    for a in b["ann"]:
                        ov = min(e, a["e"]) - max(s, a["s"])
                        if ov > 0 and (best is None or ov > best[1]):
                            best = (a, ov)
                    if best and best[0]["label"] == lab and best[1] >= 0.5 * (e - s):
                        correct += 1
                        bnd.append(abs(s - best[0]["s"]) + abs(e - best[0]["e"]))
            t1.append(
                {
                    "label": lab,
                    "query_bout": qb["wav"],
                    "accepted": accepted,
                    "correct_class": correct,
                    "precision": round(correct / accepted, 4) if accepted else None,
                    "supported_validation_occurrences": n_val_occ,
                    "recall": round(correct / n_val_occ, 4) if n_val_occ else None,
                    "median_boundary_error_frames": round(float(np.median(bnd)), 1)
                    if bnd
                    else None,
                }
            )

        # ---- TEST 2: blind discovery (labels hidden; used only for scoring)
        seeds = []
        disc_ev = sorted(parts["discovery"], key=lambda b: b["wav"])
        for L in LENGTHS:
            for qi, pa in enumerate(disc_ev):
                wq, sq = std._windows(pa["z"], L, L)
                if not len(wq):
                    continue
                for ci, pb in enumerate(disc_ev):
                    if qi == ci or len(pb["z"]) < L:
                        continue
                    wc2, sc2 = std._windows(pb["z"], L, 2)
                    if not len(wc2):
                        continue
                    dm = np.sqrt(np.maximum(2 - 2 * (wq @ wc2.T), 0))
                    best, bj = dm.min(axis=1), dm.argmin(axis=1)
                    i_row = int(np.argmin(best))
                    if best[i_row] <= radius:
                        seeds.append(
                            {
                                "L": L,
                                "dist": float(best[i_row]),
                                "a": pa,
                                "a_s": sq[i_row],
                                "b": pb,
                                "b_s": sc2[bj[i_row]],
                            }
                        )
        seeds.sort(key=lambda s: s["dist"])
        fams = []
        used: set = set()
        for s in seeds[:150]:
            if s["a"]["wav"] == s["b"]["wav"]:
                continue
            key = (s["a"]["wav"], s["a_s"], s["b"]["wav"])
            if key in used:
                continue
            q = s["a"]["z"][s["a_s"] : s["a_s"] + s["L"]]
            cost, cs, ce = std.subseq_dtw(q, s["b"]["z"])
            if not np.isfinite(cost) or cost > radius:
                continue
            occ = [(s["a"], s["a_s"], s["a_s"] + s["L"]), (s["b"], cs, ce)]
            for p in disc_ev:
                if p["wav"] in {s["a"]["wav"], s["b"]["wav"]}:
                    continue
                c2, s2, e2 = std.subseq_dtw(q, p["z"])
                if np.isfinite(c2) and c2 <= radius:
                    occ.append((p, s2, e2))
            if len({o[0]["wav"] for o in occ}) < 3:
                continue
            used.add(key)
            labels = []
            for ob, a, b_ in occ:
                best = None
                for an in ob["ann"]:
                    ov = min(b_, an["e"]) - max(a, an["s"])
                    if ov > 0 and (best is None or ov > best[1]):
                        best = (an, ov)
                labels.append(best[0]["label"] if best else None)
            # all-gap families (no member overlapping any annotation) report
            # purity None — never a fake 1.0 from counting Nones
            labeled = [x for x in labels if x is not None]
            maj = max(set(labeled), key=labeled.count) if labeled else None
            fams.append(
                {
                    "exemplar": s["a"]["wav"],
                    "L": s["L"],
                    "n_events": len({o[0]["wav"] for o in occ}),
                    "majority_label": maj,
                    "purity": round(labeled.count(maj) / len(labeled), 4) if labeled else None,
                    "labels": labels[:12],
                }
            )
            if len(fams) >= 40:
                break
        cov = {f["majority_label"] for f in fams if f["majority_label"]}

        # ---- supplementary matchability reference
        supp = []
        for i in range(15):
            p = disc_ev[rng.randint(0, len(disc_ev))]
            L = LENGTHS[i % len(LENGTHS)]
            if len(p["z"]) < L + 2:
                continue
            s0 = int(rng.randint(0, len(p["z"]) - L))
            q = p["z"][s0 : s0 + L]
            hits = sum(
                1
                for pv in parts["validation"]
                if np.isfinite(std.subseq_dtw(q, pv["z"])[0])
                and std.subseq_dtw(q, pv["z"])[0] <= radius
            )
            supp.append(hits / max(len(parts["validation"]), 1))

        per_bird[bird] = {
            "release_enforcement": enforcement,
            "counts": {
                "bouts": len(loaded),
                "annotated_syllables": n_ann,
                "unsupported": n_unsup,
                "unsupported_fraction": round(n_unsup / max(n_ann, 1), 4),
                "classes": len(class_counts),
            },
            "radius_reestimated": round(radius, 4),
            "test1_natural_known_query": t1,
            "test2_blind_discovery": {
                "n_families": len(fams),
                "n_labeled_families": sum(1 for f in fams if f["purity"] is not None),
                "n_gap_only_families": sum(1 for f in fams if f["purity"] is None),
                "mean_purity_over_labeled": round(
                    float(np.mean([f["purity"] for f in fams if f["purity"] is not None])), 4
                )
                if any(f["purity"] is not None for f in fams)
                else None,
                "classes_covered": len(cov),
                "families_per_majority_label": {
                    str(k): sum(1 for f in fams if f["majority_label"] == k) for k in cov
                },
                "families": fams[:20],
            },
            "supplementary_matchability": {
                "mean": round(float(np.mean(supp or [0])), 4),
                "note": "reference only — reference accuracy, not frequency, "
                "is the primary endpoint",
            },
        }

    # across-individual distribution
    t1_all = [c for b in per_bird.values() for c in b["test1_natural_known_query"]]
    report = {
        "framing": "natural-repetition recovery against an independent "
        "annotation reference — validates discovery capability in this "
        "regime; NOT intra-call words, NOT meaning, NOT a bat-negative "
        "explanation",
        "procedure_portability": "calibration algorithm fixed, scaler and "
        "radius re-estimated per bird from its own discovery/calibration "
        "recordings; bat radius NOT transplanted",
        "per_bird": per_bird,
        "across_birds": {
            "n_birds": len(per_bird),
            "test1_precision_values": [
                c["precision"] for c in t1_all if c["precision"] is not None
            ],
            "test1_recall_values": [c["recall"] for c in t1_all if c["recall"] is not None],
            "test1_median_boundary_error_values": [
                c["median_boundary_error_frames"]
                for c in t1_all
                if c["median_boundary_error_frames"] is not None
            ],
            "test2_mean_purity_values": [
                b["test2_blind_discovery"]["mean_purity_over_labeled"] for b in per_bird.values()
            ],
        },
        "outcome_table_mapping": "both tests recover annotated units -> "
        "proceed to marmoset within-call study; T1 only -> focus on "
        "candidate selection/family construction; neither -> natural-"
        "variation sensitivity limits; per-bird heterogeneity -> report it",
    }
    out = ROOT / "outputs" / "bengalese_natural_benchmark.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["across_birds"], indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
