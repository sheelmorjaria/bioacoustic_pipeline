#!/usr/bin/env python3
"""Marmoset replication of the POSITIVE older-history forecasting contrast.

SPEC (per review): a SEPARATE study replicating only the supporting
finding — do observations at 40/80/120/200 ms before the current state
improve a 40 ms-ahead forecast beyond the two-state [s_t, s_{t-10ms}]
baseline in InfantMarmosetsVox? NOT a context replication (no C11/C12
equivalent is claimed); the context ladder is not run.

STEP 1 — SUPPORT-ONLY AUDIT (runs first; NO models): contiguous-span
eligibility identical to the bat study (frames [t-20, t+4] within ONE
span; >=280 ms contiguous support). INSUFFICIENT SUPPORT IS AN
ACCEPTABLE TERMINAL OUTCOME, reported without further analysis.

STEP 2 — only if support suffices: two-arm comparison (short vs
shared-longer), species-specific scaler and ridge fitting (training
material only), file = recording-group identity (70/30 block split,
train-side validation for alpha), identical eligible positions across
arms, paired block-clustered bootstrap shared across arms. Loss and
aggregation definitions bound to the artifact, matching the bat study.

Material: cache/marmoset_traj.npz (the crosstaxon frozen trajectory
store; 203 calls / 10 callers / 500-frame cap; 16 kHz, 40 ms / 10 ms
framing). Release contract enforced.

Usage:
    python scripts/marmoset_history_replication.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from history_context_forecasting import ALPHAS, H_MS, LAGS, REQUIRED_FRAMES

BOOT_B = 2000
TRAIN_FRAC = 0.7
VAL_FRAC = 0.25
SEED = 0
MIN_TOTAL_POSITIONS = 2000  # declared sufficiency bar for STEP 2


MAX_CALLS = 25000  # full eligible set (all calls >= 0.6 s)
MIN_CALL_S = 0.6  # >=280 ms support needs headroom; declared filter


def _extract_call(job):
    """Fingerprint trajectory for one annotated call interval (16 kHz)."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.utils.audio import load_audio

    path, start, end, name, caller = job
    audio, sr = load_audio(str(path), target_sr=16000)
    a, b = int(start * sr), min(int(end * sr), len(audio))
    seg = audio[a:b]
    win, hop = int(sr * 0.04), int(sr * 0.01)
    rows, i = [], 0
    while i * hop + win <= len(seg) and len(rows) < 600:
        rows.append(
            np.asarray(
                afp.extract(seg[i * hop : i * hop + win], sr, species="marmoset").combined,
                dtype=np.float64,
            )
        )
        i += 1
    if len(rows) < REQUIRED_FRAMES:
        return None
    return name, caller, np.stack(rows)


def _load_calls():
    """FULL-corpus support audit. The frozen 203-call cache was a
    40-files-per-caller sample built for another purpose; letting its caps
    decide would understate the corpus. Calls >= 0.6 s, bounded cap,
    parallel extraction."""
    import csv
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        contiguous_spans,
    )

    data = ROOT / "data" / "alp" / "infant_marmosets_vox" / "data"
    rows = list(csv.DictReader(open(ROOT / "data" / "alp" / "infant_marmosets_vox" / "labels.csv")))
    jobs = []
    for r in rows:
        try:
            start, end = float(r["start"]), float(r["end"])
        except ValueError:
            continue
        if end - start < MIN_CALL_S:
            continue
        # CSV paths are 'data/audio_44k/...'; mirror root is .../data
        rel = r["path"].split("/", 1)[1] if r["path"].startswith("data/") else r["path"]
        name = rel.replace("audio_44k", "audio_16k")
        p = data / name
        if not p.exists():
            continue
        jobs.append((p, start, end, name, int(r["callerID"])))
    jobs.sort(key=lambda j: j[3])
    jobs = jobs[:MAX_CALLS]
    print(f"extraction targets: {len(jobs)} calls >= {MIN_CALL_S}s", flush=True)
    with ProcessPoolExecutor(max_workers=24) as ex:
        results = [r for r in ex.map(_extract_call, jobs, chunksize=8) if r]
    # CONTRACT FIX: select the SUPPORTED 12-feature subset (schema v2),
    # as the frozen spec requires — the first implementation ran the full
    # 44-D vector, producing pathological standardized losses (11.5) and
    # a misaligned diagnostic; caught before any result was reported
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )

    fidx = [_NAME_TO_IDX[n] for n in supported_features()]
    calls = []
    for name, caller, f44 in results:
        f = f44[:, fidx]
        valid = ~np.all(f == 0, axis=1)
        pos = []
        spans = contiguous_spans(valid)
        for sid in sorted(set(spans.tolist())):
            if sid < 0:
                continue
            idx = np.where(spans == sid)[0]
            if len(idx) >= REQUIRED_FRAMES:
                lo, hi = int(idx[0]), int(idx[-1])
                pos.extend(range(lo + max(LAGS), hi - H_MS + 1))
        calls.append({"file": name, "caller": caller, "frames": f, "valid": valid, "pos": pos})
    return calls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    import numpy as np

    from acoustic_dynamics_report import ridge_fit, ridge_predict
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )
    from history_context_forecasting import _arm_matrix, _design

    calls = _load_calls()
    span_stats = {
        "calls_total": len(calls),
        "calls_with_positions": sum(1 for c in calls if c["pos"]),
        "positions_total": sum(len(c["pos"]) for c in calls),
    }
    per_caller = {}
    for c in calls:
        e = per_caller.setdefault(str(c["caller"]), {"calls": 0, "positions": 0})
        e["calls"] += 1
        e["positions"] += len(c["pos"])
    print(
        json.dumps({"STEP 1 support audit": span_stats, "per_caller": per_caller}, indent=1),
        flush=True,
    )

    report = {
        "study": "marmoset replication of the POSITIVE older-history "
        "contrast ONLY (not a context replication; no ladder)",
        "step1_audit": {
            "total": span_stats,
            "per_caller": per_caller,
            "sufficiency_bar": MIN_TOTAL_POSITIONS,
            "eligible_positions": span_stats["positions_total"],
        },
    }
    if span_stats["positions_total"] < MIN_TOTAL_POSITIONS:
        report["outcome"] = (
            "INSUFFICIENT SUPPORT — fewer than "
            f"{MIN_TOTAL_POSITIONS} eligible positions; "
            "terminal outcome per the frozen spec; no "
            "models fitted"
        )
        out = ROOT / "outputs" / "marmoset_history_replication.json"
        out.write_text(json.dumps(report, indent=1))
        print(report["outcome"], flush=True)
        print(f"wrote {out}", flush=True)
        return

    # ---- STEP 2: two-arm replication (representation contract enforced
    # at entry — the failure mode this gate exists for occurred here)
    from bioacoustic_pipeline.pipeline.release_contract import assert_feature_contract

    sample = calls[0]["frames"] if calls else np.zeros((2, 12))
    assert_feature_contract(sample, context="marmoset replication fitting")
    rows = []
    for c in calls:
        track = {"z": c["frames"], "context": 11}  # constant code: ladder unused
        for t in c["pos"]:
            x, u, y, _ = _design(track, t)
            rows.append({"x": x, "u": u, "y": y, "c": 1.0, "group": c["file"]})
    files = sorted({r["group"] for r in rows})
    contract = ScientificRunContract(
        run_name="marmoset_history_replication",
        feature_schema_version=2,
        masks={
            "source_validity": MaskProvenance(
                kind="source_validity",
                produced_by="fingerprint absolute gate",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
            "feature_support": MaskProvenance(
                kind="feature_support",
                produced_by=f"contiguous spans >= {REQUIRED_FRAMES} frames",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(files, ["replication"] * len(files), files),
        eval_split={
            "provenance": "file=recording-group hash split 70/30; train-side VAL 25% for alpha",
            "unit": "file",
        },
    )
    enforcement = raise_for_science(contract)

    groups = sorted({r["group"] for r in rows})
    fit_blocks = {
        g
        for g in groups
        if int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    ev_groups = sorted(set(groups) - fit_blocks)
    rng = np.random.RandomState(SEED)
    fit_list = sorted(fit_blocks)
    rng.shuffle(fit_list)
    val_blocks = set(fit_list[: max(1, int(len(fit_list) * VAL_FRAC))])
    fit_blocks = set(fit_list) - val_blocks
    tr = [r for r in rows if r["group"] in fit_blocks]
    va = [r for r in rows if r["group"] in val_blocks]
    ev = [r for r in rows if r["group"] in ev_groups]

    def _mats(rs):
        X = np.stack([r["x"] for r in rs])
        U = np.stack([r["u"] for r in rs])
        Y = np.stack([r["y"] for r in rs])
        return X, U, Y

    Xtr, Utr, Ytr = _mats(tr)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-9
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-9
    mu_u, sd_u = Utr.mean(0), Utr.std(0) + 1e-9

    def _norm(rs):
        X, U, Y = _mats(rs)
        return ((X - mu_x) / sd_x, (U - mu_u) / sd_u, (Y - mu) / sd)

    Xtr, Utr, Yn = _norm(tr)
    Xva, Uva, Yva = _norm(va)
    Xev, Uev, Yev = _norm(ev)

    results = {}
    resid = {}
    for arm in ("short", "shared"):
        A_tr = _arm_matrix(arm, Xtr, Utr, np.ones(len(tr)))
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(A_tr, Yn, alpha)
            loss = float(
                np.sqrt(
                    (
                        (ridge_predict(w, b, _arm_matrix(arm, Xva, Uva, np.ones(len(va)))) - Yva)
                        ** 2
                    ).mean(axis=0)
                ).mean()
            )
            if loss < best[1]:
                best = (alpha, loss)
        w, b = ridge_fit(A_tr, Yn, best[0])
        pr = ridge_predict(w, b, _arm_matrix(arm, Xev, Uev, np.ones(len(ev))))
        r_ = pr - Yev
        resid[arm] = r_
        results[arm] = {"alpha": best[0], "loss": float(np.sqrt((r_**2).mean(axis=0)).mean())}
    delta = results["short"]["loss"] - results["shared"]["loss"]

    by_group: dict = {}
    for i, r in enumerate(ev):
        by_group.setdefault(r["group"], []).append(i)
    keys = sorted(by_group)
    brng = np.random.RandomState(SEED)

    def _loss(rr):
        return float(np.sqrt((rr**2).mean(axis=0)).mean())

    reps = []
    for _ in range(BOOT_B):
        pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
        idx = np.concatenate([by_group[g] for g in pick])
        reps.append(_loss(resid["short"][idx]) - _loss(resid["shared"][idx]))
    ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]

    report.update(
        {
            "outcome": "evaluated (two arms)",
            "release_enforcement": enforcement,
            "step2": {
                "n_fit": len(tr),
                "n_val": len(va),
                "n_eval": len(ev),
                "eval_files": len(ev_groups),
                "models": results,
                "delta_older_history": round(float(delta), 4),
                "bootstrap_ci": [round(c, 4) for c in ci],
                "bootstrap_details": "95% percentile, file-level blocks with "
                "replacement, B=2000, identical picks across arms, "
                "conditional on the fitted models",
                "aggregation": "per-feature RMSE of FIT-standardized targets "
                "averaged over 12 features (identical convention to the bat "
                "study)",
            },
            "framing": "replication of the POSITIVE older-history contrast "
            "only; not a context replication; applicability conditional on "
            ">=280 ms contiguous spans; no order-dependence or biological-"
            "memory claim",
        }
    )
    out = ROOT / "outputs" / "marmoset_history_replication.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["step2"], indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
