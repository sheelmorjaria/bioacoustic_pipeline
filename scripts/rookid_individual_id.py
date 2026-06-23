#!/usr/bin/env python3
"""RookID individual-ID: whole-call vs HECB sub-call in noisy outdoor corvid recordings.

The headline RookID experiment: does hierarchical sub-call structure improve individual
identification (Source) beyond whole-call acoustics? RookID has individually-identified
captive rooks in an outdoor aviary — named individuals (Feisty, Jolene, ...), 48 kHz
4-channel, noisy outdoor recordings with other birds/human/traffic. This extends the bat
emitter finding into a cleaner, individually-annotated corvid dataset.

Per call (TSV: Source/Event/Start/End): seek from 4ch WAV (mono mixdown), compute:
  1. whole-call fingerprint (44-D combined)
  2. HECB-pooled sub-call (mean of raw-NBD sub-call fingerprints, 44-D)

File-held-out split by recording date (cross-session generalization). Evaluate:
  Source V (Cramér's V), NMI, macro-F1 (LogReg train→test held-out).

Excludes Inc (unknown) / Pls (overlap) for clean single-emitter benchmark.

Run:  python scripts/rookid_individual_id.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402

DATA = Path("/mnt/c/Users/sheel/Desktop/data/RookID")
SR = 48000
_PROPOSAL = dict(floor_energy=0.04, floor_flux=0.02, debounce_ms=25.0)


def parse_tsvs():
    """Parse all RookID TSVs → list of (wav_stem, source, event, comment, start_s, end_s)."""
    calls = []
    for tsv in sorted(DATA.glob("*.tsv")):
        stem = tsv.stem
        with open(tsv) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                src = (row.get("Source") or "").strip()
                ev = (row.get("Event") or "").strip()
                try:
                    s = float(row["Start"])
                    e = float(row["End"])
                except (ValueError, KeyError):
                    continue
                if e - s < 0.08 or e - s > 3.0:
                    continue
                if src in ("Inc", "Pls", ""):
                    continue  # unknown / overlap → clean single-emitter only
                calls.append((stem, src, ev, s, e))
    return calls


def seek_audio(wav_path, start_s, end_s):
    """Seek 4ch WAV → mono mixdown segment (no full-file load)."""
    import soundfile as sf  # noqa: PLC0415

    si = int(start_s * SR)
    ei = int(end_s * SR)
    try:
        seg, _ = sf.read(str(wav_path), start=si, stop=ei, always_2d=True, dtype="float32")
        return seg.mean(axis=1) if seg.shape[1] > 1 else seg[:, 0]
    except Exception:  # noqa: BLE001
        return None


def build_representations(calls, max_calls):
    """For each call: whole-call fp + HECB-pooled sub-call fp."""
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="").combined  # noqa: E731
    recs = []
    _wav_cache = {}
    for ci, (stem, src, ev, s, e) in enumerate(calls):
        if len(recs) >= max_calls:
            break
        if stem not in _wav_cache:
            _wav_cache[stem] = DATA / f"{stem}.wav"
        wav = _wav_cache[stem]
        if not wav.exists():
            continue
        audio = seek_audio(wav, s, e)
        if audio is None or len(audio) < int(0.03 * SR):
            continue
        whole = np.asarray(fp_fn(audio, SR), dtype=np.float32)
        if not np.any(whole):
            continue
        # HECB sub-call: raw NBD proposals → fingerprint per proposal → mean
        sig = compute_signals(audio, SR, hop_ms=10.0)
        props = generate_proposals(sig, **_PROPOSAL)
        sub_fps = []
        for p in props:
            psi = max(0, int(p.time_ms / 1000 * SR) - int(0.02 * SR))
            psj = psi + int(0.04 * SR)
            pseg = audio[psi:psj]
            if len(pseg) >= int(0.012 * SR):
                pf = np.asarray(fp_fn(pseg, SR), dtype=np.float32)
                if np.any(pf):
                    sub_fps.append(pf)
        hecb_pooled = np.mean(sub_fps, axis=0) if sub_fps else np.zeros_like(whole)
        hecb_stat = (
            np.concatenate([np.mean(sub_fps, axis=0), np.std(sub_fps, axis=0)])
            if sub_fps
            else np.concatenate([np.zeros_like(whole), np.zeros_like(whole)])
        )
        recs.append(
            dict(
                stem=stem,
                source=src,
                event=ev,
                whole=whole,
                hecb=hecb_pooled,
                hecb_stat=hecb_stat.astype(np.float32),
                dur=e - s,
                n_sub=len(sub_fps),
            )
        )
        if (ci + 1) % 500 == 0:
            print(f"  {ci + 1}/{len(calls)} processed → {len(recs)} calls", flush=True)
    return recs


def evaluate(recs, split_key, rep_key, seed=0):
    """File-held-out split by recording date → Source V / NMI / macro-F1."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score, normalized_mutual_info_score  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    # group by recording date (stem prefix = date)
    dates = sorted({r[split_key][:8] for r in recs})  # YYYYMMDD
    rng = random.Random(seed)
    rng.shuffle(dates)
    split = int(0.7 * len(dates))
    train_dates = set(dates[:split])
    train = [r for r in recs if r[split_key][:8] in train_dates]
    test = [r for r in recs if r[split_key][:8] not in train_dates]
    if not train or not test:
        return None

    Xtr = np.asarray([r[rep_key] for r in train], dtype=np.float32)
    Xte = np.asarray([r[rep_key] for r in test], dtype=np.float32)
    ytr = [r["source"] for r in train]
    yte = [r["source"] for r in test]
    common = set(ytr) & set(yte)
    # filter to sources present in both
    keep_tr = [i for i, y in enumerate(ytr) if y in common]
    keep_te = [i for i, y in enumerate(yte) if y in common]
    if len(keep_tr) < 20 or len(keep_te) < 10 or len(common) < 2:
        return None
    Xtr, ytr = Xtr[keep_tr], [ytr[i] for i in keep_tr]
    Xte, yte = Xte[keep_te], [yte[i] for i in keep_te]

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0).fit(
        sc.transform(Xtr), ytr
    )
    pred = clf.predict(sc.transform(Xte))
    macro_f1 = f1_score(yte, pred, average="macro")
    nmi = normalized_mutual_info_score(yte, pred)

    # Source V: k-means → Cramér's V on test
    from scipy.stats import chi2_contingency  # noqa: PLC0415
    from sklearn.cluster import KMeans  # noqa: PLC0415

    k = min(len(common), 20)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(sc.transform(Xte))
    ulabels = sorted(set(yte))
    table = np.zeros((k, len(ulabels)))
    for p, y in zip(km.labels_, yte):
        table[p, ulabels.index(y)] += 1
    try:
        chi2, _, _, _ = chi2_contingency(table)
        V = min(float(np.sqrt(chi2 / (table.sum() * (min(k, len(ulabels)) - 1)))), 1.0)
    except Exception:  # noqa: BLE001
        V = 0.0
    return dict(
        V=V, nmi=nmi, macro_f1=macro_f1, n_test=len(yte), n_sources=len(common), n_train=len(ytr)
    )


def main():
    ap = argparse.ArgumentParser(description="RookID individual-ID")
    ap.add_argument("--max-calls", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("parsing RookID TSVs...", flush=True)
    calls = parse_tsvs()
    sources = Counter(c[1] for c in calls)
    events = Counter(c[2] for c in calls)
    dates = sorted({c[0][:8] for c in calls})
    print(
        f"  {len(calls)} calls (excl Inc/Pls), {len(sources)} sources, "
        f"{len(events)} events, {len(dates)} recording dates",
        flush=True,
    )
    print(f"  sources: {sources.most_common(15)}", flush=True)
    print(f"  top events: {events.most_common(10)}", flush=True)

    print(f"\nbuilding representations (max {args.max_calls} calls)...", flush=True)
    recs = build_representations(calls, args.max_calls)
    print(f"  {len(recs)} calls with fingerprints", flush=True)

    print(f"\n{'=' * 80}")
    print(f"ROOKID INDIVIDUAL-ID — file-held-out by recording date\n{'=' * 80}")
    print(
        f"  {'representation':22} {'Source V':>8} {'NMI':>6} {'macroF1':>7} "
        f"{'#train':>6} {'#test':>5} {'#src':>4}"
    )
    print("  " + "-" * 64)
    for name, key in [
        ("whole-call fp (44-D)", "whole"),
        ("HECB-pooled mean (44-D)", "hecb"),
        ("HECB-pooled stat (88-D)", "hecb_stat"),
    ]:
        ev = evaluate(recs, "stem", key, args.seed)
        if ev:
            print(
                f"  {name:22} {ev['V']:8.3f} {ev['nmi']:6.3f} {ev['macro_f1']:7.3f} "
                f"{ev['n_train']:6d} {ev['n_test']:5d} {ev['n_sources']:4d}"
            )
        else:
            print(f"  {name:22} (insufficient data for held-out)")

    print("\n  split: 70% recording dates → train, 30% → test (cross-session held-out)")
    print("  Sources present in both splits only. Inc/Pls excluded.")
    print("\nROOKID INDIVIDUAL-ID DONE")


if __name__ == "__main__":
    main()
