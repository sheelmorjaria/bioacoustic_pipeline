#!/usr/bin/env python3
"""CAVEM event-level history comparison (advisor milestone, 2026-10-20):
does preceding vocal interaction history help predict whether the
annotated recipient vocalises within 5 seconds, beyond the current event
and its social context?

PREREGISTRATION (frozen before eval scores):
  ENDPOINT   Y_i = 1{recipient of event i vocalises within 5 s}, with the
             OBSERVABILITY condition: the 5-s follow-up must be covered by
             subsequent in-session recordings — an unobserved interval is
             NOT evidence of silence (missing follow-up is excluded from
             the risk set, never coded negative).
  RISK SET   index events with valid caller+recipient identities
             (non-zero, distinct), cached acoustic trajectory, and an
             observable window (>=1 in-session-stream file starting within
             (index_start, index_start + 6 s]; response searched over
             events with start in (index_start, index_start + 5 s];
             self-continuation cannot count since recipient != caller).
  MODELS     M0 metadata-only (caller/recipient/context/pre-action one-hots,
             log dt, log duration) — how predictable from the social
             situation alone;
             M1 = M0 + current-event acoustics (12-D summary);
             M2s = M1 + simple history summary (previous 30 s in-stream:
             n events, n by recipient, n by caller, mean log dt,
             recipient-active flag) — tests HISTORY FEATURES with the same
             linear family;
             M2g = GRU over preceding in-stream event summaries (the
             complexity-justified recurrent alternative).
  PRIMARY    delta_history = heldout_logloss(M1) - heldout_logloss(M2s)
             (cluster bootstrap by session; caller-cluster sensitivity).
  SPLIT      session-level hash 70/30; stopping on a fit-side carve.
  WINDOW     5 s only (the conservative population; 30-s exchanges are NOT
             additional positives).
INTERPRETATION: incremental predictive value of interaction history; no
turn-taking-intent, semantic, or language claims. One run.

Usage:
    python scripts/cavem_event_history_bat.py [--self-test]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data" / "egyptian_fruit_bats"
SESSION_GAP_S = 30.0
WINDOW_S = 5.0
OBS_COVER_S = 6.0
HIST_S = 30.0
SEED = 0
BOOT_B = 1000
FS = 250_000.0  # original annotation sample rate


def _build_streams():
    """FileInfo + original annotations -> per-(treatment, channel) event
    streams with sessions (reuses the audit's construction)."""
    import csv

    fi = {}
    with open(DATA / "FileInfo.csv") as fh:
        for row in csv.DictReader(fh):
            fi[int(row["FileID"])] = {
                "t": datetime.strptime(row["Recording time"], "%Y-%m-%d %H:%M:%S"),
                "ch": int(row["Recording channel"]),
                "treat": int(row["Treatment ID"]),
            }
    streams = defaultdict(list)
    with open(DATA / "Annotations_orig.csv") as fh:
        for row in csv.DictReader(fh):
            f = fi.get(int(row["FileID"]))
            if f is None:
                continue
            streams[(f["treat"], f["ch"])].append(
                {
                    "fid": int(row["FileID"]),
                    "t": f["t"],
                    "emitter": int(row["Emitter"]),
                    "addressee": int(row["Addressee"]),
                    "context": int(row["Context"]),
                    "pre": int(row["Emitter pre-vocalization action"]),
                    "dur_s": (int(row["End sample"]) - int(row["Start sample"])) / FS,
                }
            )
    events = []
    sess_id = 0
    for key, evs in streams.items():
        evs.sort(key=lambda x: x["t"])
        cur = [evs[0]]
        for e in evs[1:]:
            if (e["t"] - cur[-1]["t"]).total_seconds() > SESSION_GAP_S:
                for e2 in cur:
                    e2["session"] = sess_id
                events.extend(cur)
                sess_id += 1
                cur = [e]
            else:
                cur.append(e)
        for e2 in cur:
            e2["session"] = sess_id
        events.extend(cur)
        sess_id += 1
    return events


def _acoustic_cache():
    """12-D trajectory summaries for cached dyadic files (frozen set)."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import parse_annotations_csv

    audit = json.loads((ROOT / "outputs/history_support_audit.json").read_text())
    frozen = {
        (p["caller"], p["recipient"])
        for p in audit["support_audit"]["dyadic_cells_frozen"]["pairs"].values()
    }
    meta = {a.file_name: a for a in parse_annotations_csv(DATA / "annotations.csv")}
    fidx = [_NAME_TO_IDX[n] for n in supported_features()]
    out = {}
    seen = set()
    for cache in (
        ROOT / "cache/stage2/bat_c11c12_traj.npz",
        ROOT / "cache/stage2/confirm_dyn_cv_traj.npz",
    ):
        if not cache.exists():
            continue
        print(f"loading acoustic cache: {cache.name}", flush=True)
        with np.load(cache, allow_pickle=False) as d:
            names = list(d["names"])
            # NPZ members decompress on every access; materialize once per cache.
            offsets = d["offsets"]
            frames = d["frames"]
            # local names are 0.wav.. aligned to original order; index by row
            for i, nm in enumerate(names):
                if nm in seen:
                    continue
                seen.add(nm)
                z = frames[offsets[i] : offsets[i + 1]][:, fidx].astype(np.float64)
                valid = ~np.all(z == 0, axis=1)
                if valid.sum() >= 10:
                    vv = z[valid]
                    out[nm] = np.concatenate([vv.mean(0), vv.std(0)])
            del frames
        print(f"acoustic summaries available: {len(out)}", flush=True)
    return out, meta, frozen


def _risk_set(events, acoustic):
    """Index events with valid identities, cached acoustics, observable
    follow-up; Y from recipient vocalisation within the window."""
    by_session = defaultdict(list)
    for e in events:
        by_session[e["session"]].append(e)
    rows = []
    n_unobservable = n_no_acoustics = n_identity = 0
    for sid, evs in by_session.items():
        evs = sorted(evs, key=lambda x: x["t"])
        for i, e in enumerate(evs):
            if not (e["emitter"] > 0 and e["addressee"] > 0 and e["emitter"] != e["addressee"]):
                n_identity += 1
                continue
            # local-name alignment: the audit verified row order; acoustic
            # cache is keyed by local name == row index of the ORIGINAL csv
            key = f"{e['row_index']}.wav"
            summ = acoustic.get(key)
            if summ is None:
                n_no_acoustics += 1
                continue
            t0 = e["t"]
            later = [x for x in evs[i + 1 :]]
            observable = any(0 < (x["t"] - t0).total_seconds() <= OBS_COVER_S for x in later)
            if not observable:
                n_unobservable += 1
                continue
            y = int(
                any(
                    0 < (x["t"] - t0).total_seconds() <= WINDOW_S and x["emitter"] == e["addressee"]
                    for x in later
                )
            )
            hist = [x for x in evs[:i] if 0 < (t0 - x["t"]).total_seconds() <= HIST_S]
            rows.append({"e": e, "summ": summ, "y": y, "hist": hist, "session": sid})
    return rows, {
        "identity_excluded": n_identity,
        "no_acoustics": n_no_acoustics,
        "unobservable": n_unobservable,
    }


def _features(row, callers, contexts):
    import numpy as np

    e = row["e"]
    feat = []
    # M0 metadata
    cl = [0.0] * (len(callers) + 1)
    cl[min(callers.get(e["emitter"], len(callers)), len(callers))] = 1.0
    rc = [0.0] * (len(callers) + 1)
    rc[min(callers.get(e["addressee"], len(callers)), len(callers))] = 1.0
    ctx = [0.0] * (len(contexts) + 1)
    ctx[min(contexts.get(e["context"], len(contexts)), len(contexts))] = 1.0
    pre = [0.0, 0.0]
    if e["pre"] in (1, 2):
        pre[e["pre"] - 1] = 1.0
    dt = row.get("dt_prev", 0.0)
    feat = cl + rc + ctx + pre + [np.log1p(dt), np.log1p(e["dur_s"])]
    m0 = np.array(feat, float)
    m1 = np.concatenate([m0, row["summ"]])
    h = row["hist"]
    by_rec = sum(1 for x in h if x["emitter"] == e["addressee"])
    by_cal = sum(1 for x in h if x["emitter"] == e["emitter"])
    dts = [(row["e"]["t"] - x["t"]).total_seconds() for x in h]
    m2s = np.concatenate(
        [
            m1,
            [
                len(h),
                by_rec,
                by_cal,
                float(np.mean(np.log1p(dts))) if dts else 0.0,
                1.0 if by_rec else 0.0,
            ],
        ]
    )
    return m0, m1, m2s


def _logloss(p, y):
    import numpy as np

    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _boot_delta(ll_a, ll_b, clusters, b=BOOT_B, seed=SEED):
    import numpy as np

    rng = np.random.RandomState(seed)
    uc = sorted(set(clusters))
    idx_of = {c: np.where(np.array(clusters) == c)[0] for c in uc}
    A, B = np.array(ll_a), np.array(ll_b)
    diffs = []
    for _ in range(b):
        cs = rng.choice(len(uc), len(uc))
        ii = np.concatenate([idx_of[uc[c]] for c in cs])
        diffs.append(A[ii].mean() - B[ii].mean())
    return {
        "point": round(float(A.mean() - B.mean()), 4),
        "ci": [
            round(float(np.percentile(diffs, 2.5)), 4),
            round(float(np.percentile(diffs, 97.5)), 4),
        ],
    }


def _self_test() -> None:
    import numpy as np

    rng = np.random.RandomState(0)
    # two sessions; history carries the signal in session 1 style
    rows = []
    for s in range(20):
        evs = []
        for i in range(10):
            evs.append(
                {
                    "fid": s * 10 + i,
                    "t": datetime(2020, 1, 1, 0, s, (i * 2) % 60),
                    "emitter": 1 + (i % 2),
                    "addressee": 2 - (i % 2) if i % 2 == 0 else 1 + (i % 2),
                    "context": 5,
                    "pre": 2,
                    "dur_s": 0.05,
                    "session": s,
                }
            )
        for i, e in enumerate(evs):
            rows.append(
                {
                    "e": e,
                    "summ": rng.randn(24) * 0.1,
                    "y": int(i > 0 and i % 3 == 0),
                    "hist": evs[:i],
                    "session": s,
                    "dt_prev": 2.0 if i else 0.0,
                }
            )
    callers = {1: 0, 2: 1}
    contexts = {5: 0}
    feats = [_features(r, callers, contexts) for r in rows]
    assert all(len(f[2]) > len(f[1]) > len(f[0]) for f in feats)
    p = np.clip(
        0.3 + 0.1 * np.zeros(len(rows)) + 0.2 * np.array([r["y"] for r in rows]), 0.05, 0.95
    )
    ll = _logloss(p, np.array([r["y"] for r in rows]))
    assert 0 < ll < 1
    print("SELF-TEST OK (features, logloss, shapes)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    events = _build_streams()
    # attach row index (original csv order) for the acoustic-cache key
    import csv as _csv

    with open(DATA / "Annotations_orig.csv") as fh:
        for i, row in enumerate(_csv.DictReader(fh)):
            pass
    acoustic, _meta, _frozen = _acoustic_cache()
    # map fid -> local name via row order: local i.wav <-> original row i
    fid_by_row = {}
    with open(DATA / "Annotations_orig.csv") as fh:
        for i, row in enumerate(_csv.DictReader(fh)):
            fid_by_row[int(row["FileID"])] = i
    for e in events:
        e["row_index"] = fid_by_row[e["fid"]]

    rows, excl = _risk_set(events, {f"{e['row_index']}.wav": None for e in []} or acoustic)
    print(f"risk set: {len(rows)} index events (excluded: {excl})", flush=True)
    ys = np.array([r["y"] for r in rows])
    print(f"positives (recipient within {WINDOW_S}s): {int(ys.sum())} / {len(rows)}", flush=True)

    # dt_prev per row
    for r in rows:
        r["dt_prev"] = (
            (r["e"]["t"] - max((x["t"] for x in r["hist"]), default=r["e"]["t"])).total_seconds()
            if r["hist"]
            else 0.0
        )

    callers = {c: i for i, c in enumerate(sorted({r["e"]["emitter"] for r in rows})[:20])}
    contexts = {c: i for i, c in enumerate(sorted({r["e"]["context"] for r in rows})[:12])}

    sessions = sorted({r["session"] for r in rows})
    part = {}
    for s in sessions:
        h = int(hashlib.sha1(f"{SEED}|{s}".encode()).hexdigest()[:8], 16) % 100
        part[s] = "eval" if h >= 70 else "fit"
    fit_rows = [r for r in rows if part[r["session"]] == "fit"]
    eval_rows = [r for r in rows if part[r["session"]] == "eval"]

    def _arm(key):
        Xf = np.stack(
            [_features(r, callers, contexts)[{"m0": 0, "m1": 1, "m2s": 2}[key]] for r in fit_rows]
        )
        Xe = np.stack(
            [_features(r, callers, contexts)[{"m0": 0, "m1": 1, "m2s": 2}[key]] for r in eval_rows]
        )
        yf = np.array([r["y"] for r in fit_rows])
        np.array([r["y"] for r in eval_rows])
        sc = StandardScaler().fit(Xf)
        clf = LogisticRegression(max_iter=3000, C=0.5, random_state=SEED).fit(sc.transform(Xf), yf)
        p = clf.predict_proba(sc.transform(Xe))[:, 1]
        return p

    preds = {k: _arm(k) for k in ("m0", "m1", "m2s")}
    ye = np.array([r["y"] for r in eval_rows])
    sess_e = [r["session"] for r in eval_rows]
    callers_e = [r["e"]["emitter"] for r in eval_rows]
    lls = {k: round(_logloss(p, ye), 4) for k, p in preds.items()}
    delta_hist = _boot_delta(
        [0.0 if abs(preds["m1"][i] - ye[i]) < 0 else 0.0 for i in range(len(ye))],
        [0.0] * len(ye),
        sess_e,
    )  # placeholder replaced below
    per_row_ll = {}
    for k, p in preds.items():
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        per_row_ll[k] = -(ye * np.log(pc) + (1 - ye) * np.log(1 - pc))
    delta_hist = _boot_delta(per_row_ll["m1"], per_row_ll["m2s"], sess_e)
    delta_acoustic = _boot_delta(per_row_ll["m0"], per_row_ll["m1"], sess_e)
    delta_hist_caller = _boot_delta(per_row_ll["m1"], per_row_ll["m2s"], callers_e)

    out = {
        "study": "CAVEM event-level history comparison (first scientific "
        "evaluation; advisor milestone 2026-10-20)",
        "endpoint": f"Y = recipient vocalises within {WINDOW_S}s, observable "
        "windows only (missing follow-up EXCLUDED, never negative)",
        "risk_set": {"n_index": len(rows), "n_positive": int(ys.sum()), "excluded": excl},
        "models": "M0 metadata / M1 +acoustics / M2s +history summary "
        "(logistic, C=0.5); GRU deferred until the linear history test is "
        "positive (complexity discipline with ~small positive count)",
        "heldout_logloss": lls,
        "n_eval": len(eval_rows),
        "delta_acoustic_m1_vs_m0": delta_acoustic,
        "delta_history_m2s_vs_m1": delta_hist,
        "delta_history_caller_clustered_sensitivity": delta_hist_caller,
        "decision_rule": "history helps if delta_history point > 0 with CI "
        "excluding 0 (log-loss improvement on held-out, session-clustered)",
        "interpretation": "incremental predictive value of interaction "
        "history only; no turn-taking-intent/semantic claims; 30-s "
        "exchanges NOT used as positives",
        "stop_rule": "one run",
    }
    (ROOT / "outputs" / "cavem_event_history_bat.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {k: out[k] for k in ("risk_set", "heldout_logloss", "delta_history_m2s_vs_m1")},
            indent=1,
        ),
        flush=True,
    )
    print("wrote outputs/cavem_event_history_bat.json", flush=True)


if __name__ == "__main__":
    main()
