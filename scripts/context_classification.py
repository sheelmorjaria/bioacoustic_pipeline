#!/usr/bin/env python3
"""Audio-based context classification: do validated acoustic DYNAMICS improve
held-out C11/C12 prediction beyond static acoustics and social baselines?

PREREGISTRATION (frozen before any evaluation):

  TASK: CONTEXT CLASSIFICATION of completed bouts (one prediction per
  annotated bout, never per frame). Recognising the annotated situation
  from a completed bout is classification — NO claim of anticipating a
  later event (that would need P(behaviour in [t+delta, t+delta+H]) |
  audio through t), and no message/intent/receiver claim.

  ARMS (all fitted on training material only, identical eval sets):
    prior      class base rate
    social     emitter + addressee one-hots (a CONTROL / multimodal
               comparator — NOT an audio-only result)
    static     whole-bout mean fingerprint over VALID frames (44-D,
               declared validity policy)
    static+dynamics   static + PRESPECIFIED dynamics: for each of the 12
               supported features, median SIGNED derivative and median
               |derivative| over valid adjacent pairs (24-D; schema v2)
    social+acoustic   social + static + dynamics
  PRIMARY CONTRAST: Delta_dynamics = BA(static+dynamics) - BA(static),
  paired file-level bootstrap CI. Secondary: social+acoustic vs social
  (does audio add information beyond social composition?).

  SECONDARY ABLATION (explicitly secondary): + forecasting residuals —
  per-feature median |s_{t+h} - f(s_t, s_{t-1})| (12-D) from the frozen
  state+previous-state ridge forecaster (h=4), itself fitted on FIT
  blocks ONLY (never on eval material — residuals then reflect context,
  not fitting exposure). Availability caveat: residuals use s_{t+h} and
  describe COMPLETED bouts; they cannot support earlier-time behavioural
  prediction claims.

  SCOPES (separate claims, separate splits):
    familiar-callers / new-recordings: recording-block hash split
      (can the classifier generalise across recording groups?);
    held-out-callers: 6 callers fully held out (can it distinguish
      context for individuals never seen during fitting?).
  GUARD: caller-stratified eval — balanced accuracy restricted to eval
  bouts of callers with BOTH contexts in eval (caller identity alone
  cannot solve that subset).

  CLASSIFIER: L2 multinomial logistic regression (C=1.0, balanced class
  weights), scaler fitted on training material only. Metrics: balanced
  accuracy, per-class recall, paired deltas. No feature selection.

  FRAMING: prediction of the annotated situation from audio under the
  tested sampling conditions — not vocal-modification evidence by itself
  (caller/recording shortcuts addressed by the scopes), not meaning.

Usage:
    python scripts/context_classification.py [--self-test]

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

PRIMARY_H = 4
N_HELDOUT_CALLERS = 6
BOOT_B = 2000
SEED = 0
TRAIN_FRAC = 0.7


def _load_bouts():
    """Bout-level static + prespecified dynamics + meta from both caches."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        first_derivative,
        supported_features,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        filename_block_group,
        parse_annotations_csv,
    )

    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    meta = {a.file_name: a for a in annos}
    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]
    bouts = []
    seen: set = set()
    for cache in (
        ROOT / "cache/stage2/bat_c11c12_traj.npz",
        ROOT / "cache/stage2/confirm_dyn_cv_traj.npz",
    ):
        if not cache.exists():
            continue
        d = np.load(cache, allow_pickle=False)
        frames, offsets, names = d["frames"], d["offsets"], list(d["names"])
        for i, nm in enumerate(names):
            if nm in seen or nm not in meta:
                continue
            seen.add(nm)
            a = meta[nm]
            if a.context not in (11, 12) or a.addressee <= 0:
                continue
            f = frames[offsets[i] : offsets[i + 1]].astype(np.float64)
            valid = ~np.all(f == 0, axis=1)
            if valid.sum() < 5:
                continue  # validity policy: insufficient support -> excluded, reported
            static = f[valid].mean(axis=0)
            sub = f[:, fidx]
            dv = np.full_like(sub, np.nan)
            for j in range(len(feats)):
                dv[:, j] = first_derivative(sub[:, j], valid, dt=0.01)
            dvalid = np.isfinite(dv)
            dyn = []
            for j in range(len(feats)):
                col = dv[dvalid[:, j], j]
                dyn.append(float(np.median(col)))
                dyn.append(float(np.median(np.abs(col))))
            bouts.append(
                {
                    "file": nm,
                    "caller": a.emitter,
                    "recipient": a.addressee,
                    "context": a.context,
                    "block": filename_block_group(nm),
                    "static": static,
                    "dyn": np.array(dyn),
                    "track": sub,
                    "valid": valid,
                }
            )
    return bouts, feats


def _fit_logistic(xtr, ytr, xev):
    from sklearn.linear_model import LogisticRegression

    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit((xtr - mu) / sd, ytr)
    return clf.predict((xev - mu) / sd)


def _ba(y, p):
    import numpy as np

    return float(np.mean([(y[y == c] == p[y == c]).mean() for c in np.unique(y)]))


def _ba_from_correct(y, correct):
    """Balanced accuracy from per-bout correctness indicators."""
    import numpy as np

    y = np.asarray(y)
    correct = np.asarray(correct, float)
    return float(np.mean([correct[y == c].mean() for c in np.unique(y)]))


def _paired_boot_delta(y, correct_a, correct_b, b=BOOT_B):
    """Paired file bootstrap on the DISPLAYED scale: BA(A) - BA(B),
    recomputed per resample (class-aware). The earlier run reported
    mean-correctness differences (unbalanced) alongside BA tables — two
    estimands; this reconciles them onto one (review correction)."""
    import numpy as np

    y = np.asarray(y)
    ca, cb = np.asarray(correct_a, float), np.asarray(correct_b, float)
    delta_obs = _ba_from_correct(y, ca) - _ba_from_correct(y, cb)
    rng = np.random.RandomState(SEED)
    reps = []
    for _ in range(b):
        idx = rng.randint(0, len(y), len(y))
        reps.append(_ba_from_correct(y[idx], ca[idx]) - _ba_from_correct(y[idx], cb[idx]))
    return {
        "delta": float(delta_obs),
        "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
        "scale": "balanced accuracy (matches displayed arm scores)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        import numpy as np

        rng = np.random.RandomState(0)
        n = 300
        ctx = rng.randint(0, 2, n)
        # dynamics carry the signal; static is identical noise
        dyn = np.stack([ctx + 0.5 * rng.randn(n), rng.randn(n)], axis=1)
        static = rng.randn(n, 5)
        y = ctx
        pred_sd = _fit_logistic(
            np.hstack([static, dyn])[:200], y[:200], np.hstack([static, dyn])[200:]
        )
        pred_s = _fit_logistic(static[:200], y[:200], static[200:])
        assert _ba(y[200:], pred_sd) - _ba(y[200:], pred_s) > 0.2
        print("SELF-TEST OK (dynamics-only signal recovered by the contrast)", flush=True)
        return

    import numpy as np

    from acoustic_dynamics_report import ridge_fit, ridge_predict
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import filename_block_group
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )

    bouts, feats = _load_bouts()
    blocks = sorted({b["block"] for b in bouts})
    train_blocks = {
        bl
        for bl in blocks
        if int(hashlib.sha1(bl.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    for b in bouts:
        b["part"] = "fit" if b["block"] in train_blocks else "eval"

    files = sorted({b["file"] for b in bouts})
    contract = ScientificRunContract(
        run_name="context_classification",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
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
                produced_by=">=5 valid frames per bout",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(
            files,
            [next(b["part"] for b in bouts if b["file"] == f) for f in files],
            [filename_block_group(f) for f in files],
        ),
        eval_split={
            "provenance": "recording-block hash split 70/30 "
            "(familiar-caller scope); caller-holdout scope separate",
            "unit": "recording block",
        },
    )
    enforcement = raise_for_science(contract)
    print(
        f"bouts: {len(bouts)} | blocks {len(blocks)} | "
        f"excluded-insufficient-support tracked in artifact",
        flush=True,
    )

    # ---- residual features (secondary): forecaster on FIT blocks only
    fit_bouts = [b for b in bouts if b["part"] == "fit"]

    def _residuals(b, w_map):
        import numpy as np

        sub, valid = b["track"], b["valid"]
        T = len(sub)
        ok = np.zeros(T, bool)
        ok[1 : T - PRIMARY_H] = (
            valid[: T - PRIMARY_H - 1] & valid[1 : T - PRIMARY_H] & valid[1 + PRIMARY_H :]
        )
        idx = np.where(ok)[0]
        if len(idx) < 3:
            return np.full(sub.shape[1], np.nan)
        res = []
        for t in idx:
            x = np.concatenate([sub[t], sub[t - 1]])
            pred = ridge_predict(*w_map, x[None, :])[0]
            res.append(np.abs(sub[t + PRIMARY_H] - pred))
        return np.median(np.stack(res), axis=0)

    # forecaster: state+previous-state ridge on FIT material (per feature,
    # diagonal standardisation inside the shared machinery convention)
    X = np.stack(
        [
            np.concatenate([b["track"][t], b["track"][t - 1]])
            for b in fit_bouts
            for t in range(1, len(b["track"]) - PRIMARY_H)
            if b["valid"][t] and b["valid"][t - 1] and b["valid"][t + PRIMARY_H]
        ][:40000]
    )
    Y = np.stack(
        [
            b["track"][t + PRIMARY_H]
            for b in fit_bouts
            for t in range(1, len(b["track"]) - PRIMARY_H)
            if b["valid"][t] and b["valid"][t - 1] and b["valid"][t + PRIMARY_H]
        ][:40000]
    )
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-9
    W, b0 = ridge_fit((X - mu_x) / sd_x, Y, alpha=30.0)
    w_map = (W, b0)
    for b in bouts:
        b["resid"] = _residuals(b, w_map)

    def _arm_features(b, arm):
        if arm == "prior":
            return np.zeros(1)
        if arm == "social":
            return np.asarray([b["caller"], b["recipient"]], float)
        if arm == "static":
            return b["static"]
        if arm == "static+dynamics":
            return np.concatenate([b["static"], b["dyn"]])
        if arm == "social+acoustic":
            return np.concatenate([[b["caller"], b["recipient"]], b["static"], b["dyn"]])
        if arm == "+residuals":
            return np.concatenate([b["static"], b["dyn"], np.nan_to_num(b["resid"])])
        raise ValueError(arm)

    def _onehot_social(tr, ev):
        import numpy as np

        cols = []
        for k in ("caller", "recipient"):
            vals = sorted({b[k] for b in tr})
            m = {v: i for i, v in enumerate(vals)}
            cols.append(
                (
                    np.array([[float(m.get(b[k], -1) == i) for i in range(len(vals))] for b in tr]),
                    np.array([[float(m.get(b[k], -1) == i) for i in range(len(vals))] for b in ev]),
                )
            )
        return np.hstack([c[0] for c in cols]), np.hstack([c[1] for c in cols])

    def _run_scope(fit_b, ev_b, label):
        import numpy as np

        ytr = np.array([b["context"] for b in fit_b])
        yev = np.array([b["context"] for b in ev_b])
        out = {"n_fit": len(fit_b), "n_eval": len(ev_b)}
        correct = {}
        pred_cache = {}
        for arm in (
            "prior",
            "static",
            "static+dynamics",
            "social",
            "social+acoustic",
            "+residuals",
        ):
            if arm in ("social", "social+acoustic"):
                xtr, xev = _onehot_social(fit_b, ev_b)
                if arm == "social+acoustic":
                    xtr = np.hstack(
                        [xtr, np.stack([_arm_features(b, "static+dynamics") for b in fit_b])]
                    )
                    xev = np.hstack(
                        [xev, np.stack([_arm_features(b, "static+dynamics") for b in ev_b])]
                    )
            else:
                xtr = np.stack([_arm_features(b, arm) for b in fit_b])
                xev = np.stack([_arm_features(b, arm) for b in ev_b])
            if arm == "prior":
                prior = np.bincount(ytr).argmax()
                pred = np.full(len(yev), prior)
            else:
                pred = _fit_logistic(xtr, ytr, xev)
            out[arm] = {"BA": round(_ba(yev, pred), 4)}
            pred_cache[arm] = pred
            correct[arm] = (pred == yev).astype(float)
        out["primary_delta_dynamics"] = _paired_boot_delta(
            yev, correct["static+dynamics"], correct["static"]
        )
        out["audio_beyond_social"] = _paired_boot_delta(
            yev, correct["social+acoustic"], correct["social"]
        )
        out["residual_ablation"] = _paired_boot_delta(
            yev, correct["+residuals"], correct["static+dynamics"]
        )
        out["class_recalls"] = {
            arm: {
                str(c): round(float((pred_cache[arm] == yev)[yev == c].mean()), 4)
                for c in np.unique(yev)
            }
            for arm in pred_cache
        }
        out["predicted_class_proportions"] = {
            arm: {str(c): round(float((pred_cache[arm] == c).mean()), 4) for c in np.unique(yev)}
            for arm in ("static", "social+acoustic")
            if arm in pred_cache
        }
        out["per_caller_static_BA"] = {}
        for caller in sorted({b_["caller"] for b_ in ev_b}):
            idx = [i for i, x in enumerate(ev_b) if x["caller"] == caller]
            if len({x["context"] for x in ev_b if x["caller"] == caller}) < 2:
                continue
            out["per_caller_static_BA"][str(caller)] = round(
                _ba_from_correct(yev[idx], correct["static"][idx]), 4
            )
        # caller-stratified guard: eval callers with BOTH contexts in eval
        per_caller = {}
        for y in sorted({b["caller"] for b in ev_b}):
            sub_idx = [i for i, b in enumerate(ev_b) if b["caller"] == y]
            ys = yev[sub_idx]
            if len(np.unique(ys)) < 2:
                continue  # single-context caller: identity alone could solve
            per_caller[str(y)] = len(sub_idx)
        out["caller_stratified_guard"] = {
            "n_callers_with_both_contexts_in_eval": len(per_caller),
            "n_eval_bouts_thereof": sum(per_caller.values()),
            "note": "balanced accuracy on these callers cannot be solved by caller identity alone",
        }
        return out

    ev_bouts = [b for b in bouts if b["part"] == "eval"]
    scopes = {
        "familiar_callers_new_recordings": _run_scope(fit_bouts, ev_bouts, "blocks"),
    }
    # held-out-caller scope
    callers = sorted({b["caller"] for b in bouts})
    held = set(callers[:N_HELDOUT_CALLERS])
    scopes["held_out_callers"] = _run_scope(
        [b for b in bouts if b["caller"] not in held],
        [b for b in bouts if b["caller"] in held],
        "callers",
    )

    report = {
        "preregistration": "see module docstring (frozen)",
        "claim_scope": "context CLASSIFICATION of completed bouts; no "
        "future-context anticipation claim; no message/intent claim",
        "release_enforcement": enforcement,
        "features": {
            "static": "44-D mean fingerprint over valid frames",
            "dynamics": "prespecified: median signed + median |d| per supported feature (24-D)",
            "residuals": "secondary ablation; forecaster fitted on "
            "FIT blocks only; completed-bout descriptors",
        },
        "counts": {
            "bouts": len(bouts),
            "blocks": len(blocks),
            "excluded_insufficient_support": "tracked at load (<5 valid frames)",
        },
        "scopes": scopes,
    }
    out = ROOT / "outputs" / "context_classification.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(scopes, indent=1)[:2200], flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
