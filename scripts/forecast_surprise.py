#!/usr/bin/env python3
"""Forecast-surprise diagnostic stream + natural-boundary detection study.

PART 1 — AUDITABLE DIAGNOSTIC STREAM: the validated immediate-history
forecaster emits, per supported target position,
    predicted_state, observed_state, residual_vector,
    aggregate_surprise A_t = mean_j (r_j / sigma_j)^2   [TRAINING-fitted
    sigma_j], valid_support, source_interval, model_version.
Naming contract: 'acoustic deviation' / 'forecast surprise' — never
stress, intent, novelty-to-animal, or semantic surprise. A large
residual is consistent with many explanations (unusual transition,
recording condition, source overlap, measurement failure, biological
change); the stream FLAGS for review, it does not diagnose.

PART 2 — THE ONE FURTHER EXPERIMENT (review-chosen): can local
forecast-surprise improve detection of INDEPENDENTLY ANNOTATED natural
acoustic boundaries (Bengalese syllable onsets/offsets from the
Koumura annotations)? Arms on identical audio:
    energy-change  — |d log-RMS/dt| peaks
    spectral-flux  — frame-to-frame spectral change peaks
    surprise       — A_t from the frozen forecaster (fit on discovery
                     material only; eval bouts untouched by fitting)
    energy+flux+surprise — z-scored sum of the three
Primary endpoints (prespecified): boundary precision / recall at a
+/-20 ms tolerance, median onset/offset error (ms), false proposals per
second. Baselines are STRONG on purpose: the question is whether
forecast surprise adds localisation signal beyond standard change
detectors. NO units, NO families, NO clustering, NO semantic claims —
a boundary proposal mechanism only; forecast-error boundaries are not
biological unit boundaries.

Usage:
    python scripts/forecast_surprise.py [--birds 0 1 2] [--stream-out]

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

SR = 16000  # Bengalese
H_FRAMES = 4
TOL_FRAMES = 2  # +/-20 ms at 10 ms hop
SEED = 0


def _surprise_stream(model, mu_x, sd_x, mu, sd, track, valid, hop_s=0.01):
    """Auditable per-position stream for one bout."""
    import numpy as np

    T = len(track)
    out = []
    w, b = model
    for t in range(1, T - H_FRAMES):
        if not (valid[t] and valid[t - 1] and valid[t + H_FRAMES]):
            continue
        x = np.concatenate(
            [(track[t] - mu_x[:12]) / sd_x[:12], (track[t - 1] - mu_x[12:]) / sd_x[12:]]
        )
        pred = (x @ w + b) * sd + mu
        obs = track[t + H_FRAMES]
        resid = obs - pred
        surprise = float(((resid / sd) ** 2).mean())
        out.append(
            {
                "frame": t,
                "time_s": round(t * hop_s, 3),
                "surprise": round(surprise, 4),
                "resid_norm": np.round(resid / sd, 3).tolist(),
                "valid_support": True,
            }
        )
    return out


def _peaks(signal, valid, min_dist_frames=5, top_k=None):
    """Indices of local maxima above the signal's median, min separation."""
    import numpy as np

    s = np.asarray(signal, float)
    thr = np.median(s[valid]) + 2.0 * np.std(s[valid])
    order = np.argsort(s)[::-1]
    picked = []
    for i in order:
        if not valid[i] or s[i] < thr:
            continue
        if all(abs(i - p) >= min_dist_frames for p in picked):
            picked.append(int(i))
        if top_k and len(picked) >= top_k:
            break
    return sorted(picked)


def _boundary_prf(proposed, true_set, tol=TOL_FRAMES):
    """Precision/recall at tolerance + median error (frames -> ms)."""
    import numpy as np

    matched_p, matched_t, errs = set(), set(), []
    for p in proposed:
        cand = [tt for tt in true_set if abs(tt - p) <= tol and tt not in matched_t]
        if cand:
            tt = min(cand, key=lambda tt: abs(tt - p))
            matched_p.add(p)
            matched_t.add(tt)
            errs.append(abs(p - tt))
    prec = len(matched_p) / max(len(proposed), 1)
    rec = len(matched_t) / max(len(true_set), 1)
    return prec, rec, (float(np.median(errs)) * 10.0 if errs else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--birds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bouts", type=int, default=90)
    ap.add_argument(
        "--stream-out",
        action="store_true",
        help="also write the auditable stream for the first "
        "eval bout per bird (inspection artifact)",
    )
    args = ap.parse_args()

    import numpy as np

    import bengalese_annotation_calibration as bac
    from acoustic_dynamics_report import ridge_fit
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        supported_features,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        assert_feature_contract,
    )

    results = {}
    stream_samples = {}
    for bird in args.birds:
        dev = bac.load_bird_dev(bird, args.bouts)
        if not dev:
            continue
        for i, b in enumerate(dev):
            b["bout_id"] = f"{b['wav']}#{i}"
        sel = [b for b in dev if b["half"] == "calibration-select"]
        ev = [b for b in dev if b["half"] == "dev-eval"]
        if len(sel) < 4 or len(ev) < 4:
            continue
        assert_feature_contract(sel[0]["track"], context="surprise fitting")

        # ---- frozen immediate-history forecaster on calibration-select
        rows_x, rows_y = [], []
        for b in sel:
            T = len(b["track"])
            for t in range(1, T - H_FRAMES):
                if b["valid"][t] and b["valid"][t - 1] and b["valid"][t + H_FRAMES]:
                    rows_x.append(np.concatenate([b["track"][t], b["track"][t - 1]]))
                    rows_y.append(b["track"][t + H_FRAMES])
        X = np.stack(rows_x)
        Y = np.stack(rows_y)
        mu, sd = Y.mean(0), Y.std(0) + 1e-9
        mu_x, sd_x = X.mean(0), X.std(0) + 1e-9
        Xn = (X - mu_x) / sd_x
        w, b0 = ridge_fit(Xn, (Y - mu) / sd, alpha=30.0)
        model = (w, b0)
        model_version = {
            "type": "ridge alpha=30 immediate-history",
            "fit": "calibration-select bouts only",
            "schema": 2,
            "features": list(supported_features()),
        }

        # ---- evaluate on dev-eval bouts: four detector arms
        arm_stats = {
            a: {"p": [], "r": [], "e": [], "fps": []}
            for a in ("energy", "flux", "surprise", "combined")
        }
        n_bouts = 0
        for b in ev:
            track, valid = b["track"], b["valid"]
            T = len(track)
            if T < 40 or valid.sum() < 30:
                continue
            n_bouts += 1
            true_on = sorted({a["s"] for a in b["ann"]})
            sorted({a["e"] for a in b["ann"]})
            if not true_on:
                continue
            # energy change |d log RMS|
            rms = np.sqrt((track**2).sum(axis=1))
            logr = np.log(rms + 1e-9)
            energy = np.abs(np.diff(logr, prepend=logr[0]))
            # spectral flux: L2 of frame-to-frame feature change (declared
            # simple baseline on the same grid)
            flux = np.linalg.norm(np.diff(track, axis=0, prepend=track[:1]), axis=1)
            # forecast surprise
            surprise = np.zeros(T)
            for t in range(1, T - H_FRAMES):
                if valid[t] and valid[t - 1] and valid[t + H_FRAMES]:
                    x = np.concatenate(
                        [(track[t] - mu_x[:12]) / sd_x[:12], (track[t - 1] - mu_x[12:]) / sd_x[12:]]
                    )
                    pred = (x @ w + b0) * sd + mu
                    surprise[t] = float((((track[t + H_FRAMES] - pred) / sd) ** 2).mean())

            def _z(s):
                m, sds = np.median(s[valid]), np.std(s[valid]) + 1e-12
                return (s - m) / sds

            combined = _z(energy) + _z(flux) + _z(surprise)
            for arm, sig in (
                ("energy", energy),
                ("flux", flux),
                ("surprise", surprise),
                ("combined", combined),
            ):
                props = _peaks(sig, valid)
                p, r, e = _boundary_prf(props, true_on)
                arm_stats[arm]["p"].append(p)
                arm_stats[arm]["r"].append(r)
                if e is not None:
                    arm_stats[arm]["e"].append(e)
                arm_stats[arm]["fps"].append(len(props) / (T * 0.01))
            if args.stream_out and bird not in stream_samples:
                stream_samples[str(bird)] = {
                    "bout": b["bout_id"],
                    "model_version": model_version,
                    "stream": _surprise_stream(model, mu_x, sd_x, mu, sd, track, valid)[:400],
                }
        results[str(bird)] = {
            "n_eval_bouts": n_bouts,
            "model": model_version,
            "arms": {
                a: {
                    "precision": round(float(np.mean(v["p"])), 4),
                    "recall": round(float(np.mean(v["r"])), 4),
                    "median_onset_error_ms": round(float(np.median(v["e"])), 1) if v["e"] else None,
                    "false_proposals_per_s": round(float(np.mean(v["fps"])), 3),
                }
                for a, v in arm_stats.items()
            },
        }
        print(
            f"[Bird{bird}] {n_bouts} eval bouts | "
            + " | ".join(
                f"{a}: P={results[str(bird)]['arms'][a]['precision']:.3f} "
                f"R={results[str(bird)]['arms'][a]['recall']:.3f}"
                for a in ("energy", "flux", "surprise", "combined")
            ),
            flush=True,
        )

    report = {
        "study": "forecast-surprise as a natural-boundary proposal signal "
        "(Bengalese annotated syllable onsets; +/-20 ms tolerance)",
        "naming_contract": "acoustic deviation / forecast surprise — never "
        "stress, intent, novelty-to-animal, or semantic surprise",
        "framing": "boundary PROPOSAL mechanism; forecast-error boundaries "
        "are not biological unit boundaries; no units/families/clustering",
        "per_bird": results,
        "endpoints": "boundary precision/recall at +/-20 ms; median onset "
        "error (ms); false proposals per second; baselines deliberately "
        "strong (energy-change, spectral-flux, combined)",
    }
    out = ROOT / "outputs" / "forecast_surprise_boundaries.json"
    out.write_text(json.dumps(report, indent=1))
    if args.stream_out:
        (ROOT / "outputs" / "forecast_surprise_stream_sample.json").write_text(
            json.dumps(stream_samples, indent=1)
        )
        print("stream sample written", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
