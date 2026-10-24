#!/usr/bin/env python3
"""Chiffchaff forecasting portability: support-only audit, then the frozen
five-arm comparison (ONE prospective avian extension; stop regardless of sign).

STEP 1 — SUPPORT-ONLY AUDIT (no models): can chiffchaff support the frozen
forecasting comparison with adequate independent recordings? Audits:
  - recording/session identity: filenames encode day + individual
    (cutted_day{N}_{PCxxxx}_{clip}); original-recording proxy = (day,
    individual); ALL clips of one recording stay in one partition;
  - FOREGROUND provenance preserved: the corpus ships fg-only WAVs (the
    bg recordings of the release are not in this mirror); nothing but fg
    material enters the vocal endpoint — recorded, not assumed;
  - contiguous ≥280 ms supported spans (frames [t-20, t+4] within ONE
    span); gaps NEVER deleted or concatenated; MIN duration filter = the
    forecasting requirement itself (0.28 s + window margin = 0.6 s
    declared headroom, same as marmoset);
  - callers/groups contributing eligible positions, and exclusions.
INSUFFICIENT SUPPORT IS AN ACCEPTABLE TERMINAL OUTCOME (no relaxed gates;
pipit is the prespecified fallback for LACK OF SUPPORT, never for an
unfavourable result).

STEP 2 — only if support adequate: five arms (training-mean reference,
persistence, current state, current+previous = PRIMARY BASELINE,
+older lags 40/80/120/200 ms = PRIMARY INTERVENTION):
    Delta_older = L(two-state) - L(two-state+older)   [positive = improvement]
with state-vs-two-state as a prespecified SECONDARY contrast. 40 ms
window / 10 ms hop / 40 ms horizon FIXED; species-specific scaler and
ridge refit on chiffchaff training material (no coefficient transfer);
alpha on train-side validation; identical eligible positions across
arms; feature-contract gate ENFORCED at fitting and scoring entry;
uncertainty = recording-cluster paired bootstrap (models fixed), with a
caller-balanced sensitivity analysis NAMED separately (weighting policy
chosen before evaluation). Population = new recording groups from
familiar callers (unseen-caller analysis is a different extension, not
mixed in). Outcome table (prespecified): improvement / recent-only /
persistence-competitive / wide-interval-inconclusive / inadequate
support — no species-difference inference from single-corpus signs.

Usage:
    python scripts/chiffchaff_forecasting_portability.py [--jobs 24]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SR = 44100
WIN_MS, HOP_MS = 40.0, 10.0
REQUIRED_FRAMES = 28
MIN_CLIP_S = 0.6
TRAIN_FRAC = 0.7
VAL_FRAC = 0.25
BOOT_B = 2000
SEED = 0
MIN_TOTAL_POSITIONS = 2000
MIN_GROUPS = 8

_FNAME = re.compile(r"cutted_day(\d+)_(PC\d+)_(\d+)\.wav")


def _recording_id(name: str) -> str:
    m = _FNAME.search(Path(name).name)
    return f"day{m.group(1)}_{m.group(2)}" if m else Path(name).stem


def _caller_id(name: str) -> str:
    m = _FNAME.search(Path(name).name)
    return m.group(2) if m else "unknown"


def _extract(job):
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.utils.audio import load_audio

    path, name = job
    audio, sr = load_audio(str(path), target_sr=SR)
    win, hop = int(sr * WIN_MS / 1000), int(sr * HOP_MS / 1000)
    rows, i = [], 0
    while i * hop + win <= len(audio) and len(rows) < 600:
        rows.append(
            np.asarray(
                afp.extract(audio[i * hop : i * hop + win], sr, species="chiffchaff").combined,
                dtype=np.float64,
            )
        )
        i += 1
    if len(rows) < REQUIRED_FRAMES:
        return None
    return name, np.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()

    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from acoustic_dynamics_report import ridge_fit, ridge_predict
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        assert_feature_contract,
        raise_for_science,
        source_manifest_from,
    )
    from history_context_forecasting import ALPHAS, H_MS, LAGS, _design

    raw = ROOT / "data" / "alp" / "chiffchaff_id" / "v0.1.0" / "raw"
    wavs = sorted((raw / "wav" / "chiffchaff-fg").glob("*.wav"))
    print(
        f"foreground WAVs found: {len(wavs)} (bg material not present in "
        "this mirror — provenance recorded, not assumed)",
        flush=True,
    )
    jobs = [(p, p.name) for p in wavs]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = [r for r in ex.map(_extract, jobs, chunksize=8) if r]
    fidx = [_NAME_TO_IDX[n] for n in supported_features()]

    # ---- audit
    clips = []
    n_short = 0
    for name, f44 in results:
        f = f44[:, fidx]
        valid = ~np.all(f == 0, axis=1)
        from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
            contiguous_spans,
        )

        pos = []
        for sid in sorted(set(contiguous_spans(valid).tolist())):
            if sid < 0:
                continue
            idx = np.where(contiguous_spans(valid) == sid)[0]
            if len(idx) >= REQUIRED_FRAMES:
                lo, hi = int(idx[0]), int(idx[-1])
                pos.extend(range(lo + max(LAGS), hi - H_MS + 1))
        if pos:
            clips.append(
                {
                    "name": name,
                    "rec": _recording_id(name),
                    "caller": _caller_id(name),
                    "frames": f,
                    "valid": valid,
                    "pos": pos,
                }
            )
        else:
            n_short += 1
    positions_total = sum(len(c["pos"]) for c in clips)
    groups = sorted({c["rec"] for c in clips})
    callers = sorted({c["caller"] for c in clips})
    per_caller = {}
    for c in clips:
        e = per_caller.setdefault(c["caller"], 0)
        per_caller[c["caller"]] = e + len(c["pos"])
    top_share = max(per_caller.values()) / max(positions_total, 1)
    print(
        json.dumps(
            {
                "clips_extracted": len(results),
                "clips_no_support": n_short,
                "clips_with_positions": len(clips),
                "positions_total": positions_total,
                "groups": len(groups),
                "callers": len(callers),
                "top_caller_position_share": round(top_share, 3),
            },
            indent=1,
        ),
        flush=True,
    )

    audit = {
        "clips_extracted": len(results),
        "clips_no_support": n_short,
        "clips_with_positions": len(clips),
        "positions_total": positions_total,
        "recording_groups": len(groups),
        "callers": len(callers),
        "per_caller_positions": {k: int(v) for k, v in sorted(per_caller.items())},
        "top_caller_share": round(top_share, 3),
        "provenance": "fg-only corpus mirror; recording proxy = (day, "
        "individual) from filenames; gaps never concatenated; "
        f"MIN_CLIP_S={MIN_CLIP_S} declared headroom",
        "sufficiency_bars": {"positions": MIN_TOTAL_POSITIONS, "groups": MIN_GROUPS},
    }
    if positions_total < MIN_TOTAL_POSITIONS or len(groups) < MIN_GROUPS:
        out = ROOT / "outputs" / "chiffchaff_forecasting_portability.json"
        out.write_text(
            json.dumps(
                {
                    "study": "chiffchaff forecasting portability — STEP 1 ONLY",
                    "audit": audit,
                    "outcome": "INSUFFICIENT SUPPORT — terminal outcome per the "
                    "frozen spec; no models fitted; pipit is the "
                    "prespecified fallback for lack of support",
                },
                indent=1,
            )
        )
        print("INSUFFICIENT SUPPORT — terminal; no models fitted", flush=True)
        print(f"wrote {out}", flush=True)
        return

    # ---- STEP 2: five arms
    assert_feature_contract(clips[0]["frames"], context="chiffchaff fitting")
    rows = []
    for c in clips:
        track = {"z": c["frames"], "context": 11}
        for t in c["pos"]:
            x, u, y, _ = _design(track, t)
            rows.append(
                {
                    "x": x,
                    "u": u,
                    "y": y,
                    "group": c["rec"],
                    "caller": c["caller"],
                    "clip": c["name"],
                }
            )
    rec_ids = sorted({r["group"] for r in rows})
    fit_recs = {
        g
        for g in rec_ids
        if int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    rng = np.random.RandomState(SEED)
    fit_list = sorted(fit_recs)
    rng.shuffle(fit_list)
    val_recs = set(fit_list[: max(1, int(len(fit_list) * VAL_FRAC))])
    fit_recs = set(fit_list) - val_recs
    tr = [r for r in rows if r["group"] in fit_recs]
    va = [r for r in rows if r["group"] in val_recs]
    ev = [r for r in rows if r["group"] not in fit_recs | val_recs]

    files = sorted({c["name"] for c in clips})
    contract = ScientificRunContract(
        run_name="chiffchaff_forecasting_portability",
        feature_schema_version=2,
        masks={
            "source_validity": MaskProvenance(
                kind="source_validity",
                produced_by="fingerprint absolute gate",
                win_ms=WIN_MS,
                hop_ms=HOP_MS,
                amplitude_stage="native waveform",
            ),
            "feature_support": MaskProvenance(
                kind="feature_support",
                produced_by=f"contiguous spans >= {REQUIRED_FRAMES} frames",
                win_ms=WIN_MS,
                hop_ms=HOP_MS,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(
            files,
            [
                "fit"
                if _recording_id(f) in fit_recs
                else ("val" if _recording_id(f) in val_recs else "eval")
                for f in files
            ],
            [_recording_id(f) for f in files],
        ),
        eval_split={
            "provenance": "recording-group (day, individual) hash "
            "split 70/30; train-side VAL 25% for alpha",
            "unit": "recording group",
        },
    )
    enforcement = raise_for_science(contract)

    Ytr = np.stack([r["y"] for r in tr])
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-9
    Xtr = np.stack([r["x"] for r in tr])
    Utr = np.stack([r["u"] for r in tr])
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-9
    mu_u, sd_u = Utr.mean(0), Utr.std(0) + 1e-9
    # scoring-entry gate: applied to the TRAJECTORY frames feeding scoring
    # (the design matrix is intentionally 24-wide — two stacked states)
    ev_clip_names = {r["clip"] for r in ev} if ev and "clip" in ev[0] else None
    if ev_clip_names:
        for c in clips:
            if c["name"] in ev_clip_names:
                assert_feature_contract(c["frames"], context="chiffchaff scoring")
                break

    def _n(rs):
        X = (np.stack([r["x"] for r in rs]) - mu_x) / sd_x
        U = (np.stack([r["u"] for r in rs]) - mu_u) / sd_u
        Y = (np.stack([r["y"] for r in rs]) - mu) / sd
        return X, U, Y

    Xtr, Utr, Yn = _n(tr)
    Xva, Uva, Yva = _n(va)
    Xev, Uev, Yev = _n(ev)
    ones_tr = np.ones(len(tr))
    ones_va = np.ones(len(va))
    ones_ev = np.ones(len(ev))

    from history_context_forecasting import _arm_matrix

    arms = {}
    resid = {}
    for arm in ("short", "shared"):
        A = _arm_matrix(arm, Xtr, Utr, ones_tr)
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(A, Yn, alpha)
            loss_v = float(
                np.sqrt(
                    ((ridge_predict(w, b, _arm_matrix(arm, Xva, Uva, ones_va)) - Yva) ** 2).mean(
                        axis=0
                    )
                ).mean()
            )
            if loss_v < best[1]:
                best = (alpha, loss_v)
        w, b = ridge_fit(A, Yn, best[0])
        pr = ridge_predict(w, b, _arm_matrix(arm, Xev, Uev, ones_ev))
        r_ = pr - Yev
        resid[arm] = r_
        arms[arm] = {"alpha": best[0], "loss": float(np.sqrt((r_**2).mean(axis=0)).mean())}
    # reference arms on the SAME eval positions
    s_t = np.stack([r["x"][:12] for r in ev])  # unnormalized current state
    pers = (s_t - mu) / sd - Yev
    arms["persistence"] = {"loss": float(np.sqrt((pers**2).mean(0)).mean())}
    arms["training_mean"] = {"loss": float(np.sqrt((Yev**2).mean(0)).mean())}
    # state-only arm
    A = Xtr[:, :12]
    best = (None, np.inf)
    for alpha in ALPHAS:
        w, b = ridge_fit(A, Yn, alpha)
        loss_v = float(np.sqrt(((ridge_predict(w, b, Xva[:, :12]) - Yva) ** 2).mean(axis=0)).mean())
        if loss_v < best[1]:
            best = (alpha, loss_v)
    w, b = ridge_fit(A, Yn, best[0])
    st = ridge_predict(w, b, Xev[:, :12]) - Yev
    resid["state"] = st
    arms["state"] = {"alpha": best[0], "loss": float(np.sqrt((st**2).mean(0)).mean())}

    def _loss(rr):
        return float(np.sqrt((rr**2).mean(axis=0)).mean())

    delta_older = arms["short"]["loss"] - arms["shared"]["loss"]
    delta_recent = arms["state"]["loss"] - arms["short"]["loss"]
    by_rec: dict = {}
    for i, r in enumerate(ev):
        by_rec.setdefault(r["group"], []).append(i)
    keys = sorted(by_rec)
    brng = np.random.RandomState(SEED)
    ci = {}
    for name, (ra, rb) in {
        "delta_older": ("short", "shared"),
        "delta_recent_secondary": ("state", "short"),
    }.items():
        reps = []
        for _ in range(BOOT_B):
            pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
            idx = np.concatenate([by_rec[g] for g in pick])
            reps.append(_loss(resid[ra][idx]) - _loss(resid[rb][idx]))
        ci[name] = [
            round(float(np.percentile(reps, 2.5)), 4),
            round(float(np.percentile(reps, 97.5)), 4),
        ]
    # caller-balanced sensitivity (NAMED separately; weighting chosen pre-eval)
    by_caller: dict = {}
    for i, r in enumerate(ev):
        by_caller.setdefault(r["caller"], []).append(i)
    cb = {
        a: round(float(np.mean([_loss(resid[a][idx]) for idx in by_caller.values()])), 4)
        for a in ("short", "shared")
    }

    out = ROOT / "outputs" / "chiffchaff_forecasting_portability.json"
    out.write_text(
        json.dumps(
            {
                "study": "chiffchaff forecasting portability (one prospective avian "
                "extension; stop regardless of sign)",
                "audit": audit,
                "release_enforcement": enforcement,
                "step2": {
                    "n_fit": len(tr),
                    "n_val": len(va),
                    "n_eval": len(ev),
                    "eval_recordings": len({r["group"] for r in ev}),
                    "eval_callers": sorted({r["caller"] for r in ev}),
                    "arms": {
                        k: (
                            {
                                "loss": round(v["loss"], 4),
                                **({"alpha": v["alpha"]} if "alpha" in v else {}),
                            }
                        )
                        for k, v in arms.items()
                    },
                    "primary_delta_older": round(float(delta_older), 4),
                    "primary_ci": ci["delta_older"],
                    "secondary_delta_recent": round(float(delta_recent), 4),
                    "secondary_ci": ci["delta_recent_secondary"],
                    "caller_balanced_sensitivity": cb,
                    "bootstrap": "95% percentile, recording-cluster paired, B=2000, "
                    "models fixed; caller-balanced aggregation named separately",
                },
                "outcome_table_prespecified": [
                    "improvement -> benefit extends to this regime under the tested procedure",
                    "recent-only -> no older-lag benefit here",
                    "persistence-competitive -> fitted predictors not established "
                    "as best practical baseline",
                    "wide interval -> inconclusive at available support",
                    "insufficient support -> no acoustic-memory conclusion",
                ],
                "framing": "no species-difference inference from single-corpus "
                "signs; fg-only endpoint; gaps preserved",
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                "arms": {k: v["loss"] if "loss" in v else v for k, v in arms.items()},
                "primary_delta_older": round(float(delta_older), 4),
                "ci": ci["delta_older"],
                "secondary_delta_recent": round(float(delta_recent), 4),
                "caller_balanced": cb,
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
