#!/usr/bin/env python3
"""Experiment A: oracle-parent HECB validation on InfantMarmosetsVox.

Clean Stage-1 cross-dataset mechanistic test (separates the internal mechanism from
the CRNN detector): use GT call spans as ORACLE parent events, run HECB internal
segmentation inside them, then whole/sub/pooled recovery on calltype + caller ID.

```
GT marmoset call spans (oracle parents)
  ↓  (snap=False, gate_durations=False → parent boundaries preserved exactly)
HECB internal annotation (NBD inside span → typed classifier → internal boundaries)
  ↓
whole / sub-call / pooled representation
  ↓
calltype + caller recovery (V / NMI / probe acc / macro-F1)
```

Framing: this is "internal-boundary validation under ground-truth call spans" — NOT
end-to-end segmentation (that needs the marmoset CRNN, Experiment C).

Splits: random + twin-pair-held-out (controls for caller/family inflating call-type
recovery). Species band = 4-14 kHz (spectrally verified: marmoset calls peak 8-12 kHz).

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.representation.subcall import (  # noqa: E402
    build_subcall_representations,
    whole_sub_pooled_ablation,
)
from bioacoustic_pipeline.segmentation.hierarchical import (  # noqa: E402
    HECBDurationPriors,
    CallEvent,
    InternalSegment,
    hecb_decode,
)
from bioacoustic_pipeline.segmentation.nbd_proposal import compute_signals, generate_proposals  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402

import train_hecb_infant_marmoset as M  # reuse loaders + evidence helpers  # noqa: E402

DATA = Path("/home/sheel/InfantMarmosetsVox")
SR = 44100
HOP_MS = 10.0
PAD_MS = 30.0
_PROPOSAL = {"floor_energy": 0.04, "floor_flux": 0.02, "debounce_ms": 25.0}
MIN_INTERNAL_CALL_MS = 80.0
# Oracle mode has no CRNN → the 9 crnn-context evidence features are zeroed. Allocate the
# zero-proba array ONCE (covers up to 700s absolute time at the 10ms hop) and reuse it.
_ZERO_PROBA = np.zeros(70000)
_AUDIO_CACHE: dict = {}  # wav path → (audio, sr); a 10-min file holds many calls — load once


def _load_cached(wav):
    if wav not in _AUDIO_CACHE:
        try:
            a, _ = load_audio(str(wav), target_sr=SR)
            _AUDIO_CACHE[wav] = a
        except Exception:  # noqa: BLE001
            _AUDIO_CACHE[wav] = None
    return _AUDIO_CACHE[wav]


def _twin_of(filename: str) -> str:
    return filename.lower().split("_")[1]  # e.g. 'twin1'


def load_all_calls(n_files):
    """All GT vocalization calls across files → [(filename, start_s, end_s, calltype, caller, twin), ...]."""
    by_file = M.load_labels(DATA / "labels.csv")
    rng = random.Random(0)
    files = sorted(by_file.keys())
    rng.shuffle(files)
    out = []
    for fn in files:
        wav = M.find_wav(fn)
        if wav is None:
            continue
        for s, e, ct, caller in M.get_call_spans(by_file[fn]):
            out.append((fn, wav, s, e, ct, caller, _twin_of(fn)))
        if len(out) > 40000:
            break
    return out


def train_typed_clf(train_calls, n_train=3000):
    """Train the HECB typed classifier on GT-span-geometry labels (oracle parents).

    For each NBD proposal inside a GT span: onset (near span start) / offset (near end) /
    internal (strictly inside) / noise (outside any span). crnn_proba is unavailable in
    oracle mode → those 9 context features are zeroed (constant); the classifier learns
    from NBD scalars + geometry + patch.
    """
    rng = random.Random(0)
    pool = list(train_calls)
    rng.shuffle(pool)
    X, y = [], []
    for fn, wav, start, end, ct, caller, twin in pool[:n_train]:
        audio = _load_cached(wav)
        if audio is None:
            continue
        lo = max(0, int((start * 1000 - PAD_MS) / 1000 * SR))
        hi = min(len(audio), int((end * 1000 + PAD_MS) / 1000 * SR))
        seg = audio[lo:hi]
        if len(seg) < int(SR * 0.04):
            continue
        sig = compute_signals(seg, SR, hop_ms=HOP_MS)
        props = generate_proposals(sig, **_PROPOSAL)
        on_ms, off_ms = start * 1000, end * 1000
        for p in props:
            p.time_ms += on_ms - PAD_MS
            ev = M._extract_evidence(p, audio, SR, _ZERO_PROBA, (on_ms, off_ms, 1.0))
            X.append(ev)
            gt_starts = [on_ms]
            gt_ends = [off_ms]
            # internal if strictly inside the span (not within onset_tol of an edge)
            if abs(p.time_ms - on_ms) < 100:
                y.append(1)  # ONSET
            elif abs(p.time_ms - off_ms) < 100:
                y.append(2)  # OFFSET
            elif on_ms + 80 < p.time_ms < off_ms - 80:
                y.append(3)  # INTERNAL
            else:
                y.append(0)  # NOISE
    X, y = np.asarray(X), np.asarray(y)
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5, class_weight="balanced", random_state=0
    )
    clf.fit(X, y)
    print(
        f"  typed classifier trained on {len(X)} proposals: "
        + " ".join(
            f"{['noise', 'onset', 'offset', 'internal'][k]}={v}"
            for k, v in sorted(Counter(y.tolist()).items())
        ),
        flush=True,
    )
    return clf


def oracle_call_events(calls, clf, fp_fn, *, min_internal_ms=MIN_INTERNAL_CALL_MS):
    """Each GT call → CallEvent (oracle onset/offset) + HECB internal segments + subcall rep."""
    reps, ct_labels, caller_labels = [], [], []
    for fn, wav, start, end, ct, caller, twin in calls:
        audio = _load_cached(wav)
        if audio is None:
            continue
        on_ms, off_ms = start * 1000, end * 1000
        lo = max(0, int((on_ms - PAD_MS) / 1000 * SR))
        hi = min(len(audio), int((off_ms + PAD_MS) / 1000 * SR))
        seg = audio[lo:hi]
        if len(seg) < int(SR * 0.03):
            continue
        # HECB internal annotation inside the oracle span
        sig = compute_signals(seg, SR, hop_ms=HOP_MS)
        props = generate_proposals(sig, **_PROPOSAL)
        typed = []
        for p in props:
            p.time_ms += on_ms - PAD_MS
            ev = M._extract_evidence(p, audio, SR, _ZERO_PROBA, (on_ms, off_ms, 1.0))
            typed.append((p.time_ms, M._proba_4(clf, ev)))
        # Oracle parent event; hecb_decode snap=False, gate_durations=False → span preserved
        decoded = hecb_decode(
            [(on_ms, off_ms, 1.0)],
            typed,
            HECBDurationPriors(),
            snap=False,
            gate_durations=False,
            min_internal_call_ms=min_internal_ms,
            file_id=fn,
        )
        if not decoded:
            continue
        call = decoded[0]
        # Per-event subcall rep (label alignment guaranteed)
        sr_reps = build_subcall_representations([call], audio, SR, fp_fn)
        if sr_reps:
            reps.append(sr_reps[0])
            ct_labels.append(ct)
            caller_labels.append(caller)
    return reps, ct_labels, caller_labels


def recovery_with_macrof1(reps, labels, k, label_name):
    """whole/sub/pooled recovery: V / NMI / probe acc / macro-F1."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import f1_score  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    ablation = whole_sub_pooled_ablation(reps, labels, k=k, min_per_class=5)

    # macro-F1 per mode (whole/pooled/subcall) via stratified LogReg
    def _macrof1(key):
        if key == "whole":
            vecs = np.asarray([r.whole_fingerprint for r in reps], dtype=np.float32)
        elif key == "pooled":
            vecs = np.asarray([r.pooled_fingerprint for r in reps], dtype=np.float32)
        else:
            return None
        y = np.asarray(labels)
        keep = Counter(y.tolist())
        m = np.array([keep[c] >= 5 for c in y.tolist()])
        vecs, y = vecs[m], y[m]
        if len(set(y.tolist())) < 2 or len(vecs) < 20:
            return None
        try:
            Xtr, Xte, ytr, yte = train_test_split(
                vecs, y, test_size=0.3, random_state=0, stratify=y
            )
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
                sc.transform(Xtr), ytr
            )
            return round(float(f1_score(yte, clf.predict(sc.transform(Xte)), average="macro")), 4)
        except Exception:  # noqa: BLE001
            return None

    ablation["whole"]["macro_f1"] = _macrof1("whole")
    ablation["pooled"]["macro_f1"] = _macrof1("pooled")
    return ablation


def print_table(label_name, result):
    print(f"\n=== {label_name} recovery (whole/sub/pooled) ===")
    print(
        f"  calls: {result['n_calls']} ({result['n_with_internal']} with internal, "
        f"{result['n_subsegs']} sub-segs)"
    )
    print(f"  {'representation':22} {'Cramers V':>9} {'NMI':>7} {'probe':>7} {'macroF1':>7}")
    for name, key in [
        ("whole call", "whole"),
        ("whole + sub pool", "pooled"),
        ("sub-call", "subcall"),
    ]:
        r = result[key]
        if r is None:
            print(f"  {name:22} {'(n/a)':>9}")
            continue
        mf1 = r.get("macro_f1")
        mf1s = f"{mf1:.3f}" if mf1 is not None else "n/a"
        print(f"  {name:22} {r['V']:9.3f} {r['nmi']:7.3f} {str(r['probe']):>7} {mf1s:>7}")


def main():
    ap = argparse.ArgumentParser(description="Oracle-parent HECB validation (InfantMarmosetsVox)")
    ap.add_argument("--n-train-clf", type=int, default=2000, help="calls to train typed classifier")
    ap.add_argument("--n-eval", type=int, default=3000, help="calls per eval split")
    args = ap.parse_args()

    print("loading GT calls...", flush=True)
    all_calls = load_all_calls(n_files=999)
    print(f"  {len(all_calls)} GT calls", flush=True)

    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="marmoset").combined  # noqa: E731

    # ─── Split 1: twin-pair-held-out (strictest — unseen families) ───
    print("\n[Split: twin-pair-held-out] train twins 1-3, eval twin 5", flush=True)
    train_pool = [c for c in all_calls if c[6] in ("twin1", "twin2", "twin3")]
    eval_pool = [c for c in all_calls if c[6] == "twin5"]
    rng = random.Random(0)
    rng.shuffle(train_pool)
    rng.shuffle(eval_pool)
    clf = train_typed_clf(train_pool, n_train=args.n_train_clf)
    eval_calls = eval_pool[: args.n_eval]
    print(
        f"  eval: {len(eval_calls)} calls | calltypes: {dict(Counter(c[4] for c in eval_calls))}",
        flush=True,
    )
    reps, ct_labels, caller_labels = oracle_call_events(eval_calls, clf, fp_fn)
    print_table("calltype", recovery_with_macrof1(reps, ct_labels, 11, "calltype"))
    print_table("caller ID", recovery_with_macrof1(reps, caller_labels, 10, "caller"))

    # ─── Split 2: random (sanity, caller-inflated upper bound) ───
    print("\n[Split: random] (caller-inflated upper bound)", flush=True)
    random.Random(1).shuffle(all_calls)
    rand_train, rand_eval = (
        all_calls[: args.n_train_clf + 1000],
        all_calls[args.n_train_clf + 1000 :],
    )
    clf2 = train_typed_clf(rand_train, n_train=args.n_train_clf)
    reps2, ct2, caller2 = oracle_call_events(rand_eval[: args.n_eval], clf2, fp_fn)
    print_table("calltype", recovery_with_macrof1(reps2, ct2, 11, "calltype"))
    print_table("caller ID", recovery_with_macrof1(reps2, caller2, 10, "caller"))
    print("\nORACLE HECB MARMOSET DONE")


if __name__ == "__main__":
    main()
