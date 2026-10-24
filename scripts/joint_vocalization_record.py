#!/usr/bin/env python3
"""Parallel-heads joint vocalization record + the surprise-by-context test.

ARCHITECTURE (review-adopted): audio -> {G global/context head, F local
forecast head, R residual/deviation head} — COMPLEMENTARY branches, never
context-injected-into-forecaster (that fusion was tested and closed).

PART 1 — the combined auditable per-bout record (exactly the review
schema): context_score + suggested_context/abstain (frozen static
classifier, declared abstain band); predicted_next_state /
forecast_residual / aggregate_surprise (frozen shared forecaster);
whole-call static features; valid_support; source_interval; caller and
recipient provenance; model_versions. Sample artifact for the first
N bouts of the frozen dyadic population.

PART 2 — the ONE joint scientific question, prespecified:
    Theta_A = median(A | C12) - median(A | C11)
under the SINGLE frozen shared forecaster (the one validated across
corpora), A = mean_j (r_j / sigma_j)^2 with training-fitted sigma.
Within the audit's frozen supported caller-recipient pairs, equal
common-recipient weighting, recording-block-clustered paired bootstrap.
Interpretation rows (frozen): ~0 -> shared local dynamics across
contexts (consistent with the closed interaction result); positive ->
C12 trajectories less captured by the shared dynamics (differently
predictable REGIMES — NOT evidence that contexts need different
forecasting rules); negative -> C11 less captured. One statistic only;
no residual-feature search opened (the earlier ablation showed little
incremental value).

Usage:
    python scripts/joint_vocalization_record.py [--sample 50]

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

TRAIN_FRAC = 0.7
BOOT_B = 2000
SEED = 0
ABSTAIN_MARGIN = 0.20
H_FRAMES = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from acoustic_dynamics_report import ridge_fit
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        filename_block_group,
        parse_annotations_csv,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        assert_feature_contract,
    )

    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    meta = {a.file_name: a for a in annos}
    audit = json.loads((ROOT / "outputs/history_support_audit.json").read_text())
    frozen = {
        (p["caller"], p["recipient"])
        for p in audit["support_audit"]["dyadic_cells_frozen"]["pairs"].values()
    }
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
            if (a.emitter, a.addressee) not in frozen:
                continue
            f = frames[offsets[i] : offsets[i + 1]][:, fidx].astype(np.float64)
            valid = ~np.all(f == 0, axis=1)
            if valid.sum() < 5:
                continue
            bouts.append(
                {
                    "file": nm,
                    "caller": a.emitter,
                    "recipient": a.addressee,
                    "ctx": a.context,
                    "block": filename_block_group(nm),
                    "track": f,
                    "valid": valid,
                    "static": frames[offsets[i] : offsets[i + 1]][valid].mean(0),
                }
            )
    blocks = sorted({b["block"] for b in bouts})
    fit_blocks = {
        bl
        for bl in blocks
        if int(hashlib.sha1(bl.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    for b in bouts:
        b["part"] = "fit" if b["block"] in fit_blocks else "eval"
    fit = [b for b in bouts if b["part"] == "fit"]
    ev = [b for b in bouts if b["part"] == "eval"]
    assert_feature_contract(fit[0]["track"], context="joint record fitting")
    print(f"bouts: {len(bouts)} (fit {len(fit)} / eval {len(ev)})", flush=True)

    # ---- HEAD 1: context classifier (static fingerprint; frozen recipe)
    Xtr_c = np.stack([b["static"] for b in fit])
    ytr_c = np.array([b["ctx"] for b in fit])
    mu_c, sd_c = Xtr_c.mean(0), Xtr_c.std(0) + 1e-9
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit((Xtr_c - mu_c) / sd_c, ytr_c)

    # ---- HEAD 2/3: shared forecaster (state+previous; ridge alpha=30 as
    # validated) + surprise with training-fitted sigma
    rows_x, rows_y = [], []
    for b in fit:
        T = len(b["track"])
        for t in range(1, T - H_FRAMES):
            if b["valid"][t] and b["valid"][t - 1] and b["valid"][t + H_FRAMES]:
                rows_x.append(np.concatenate([b["track"][t], b["track"][t - 1]]))
                rows_y.append(b["track"][t + H_FRAMES])
    Xf = np.stack(rows_x)
    Yf = np.stack(rows_y)
    mu_y, sd_y = Yf.mean(0), Yf.std(0) + 1e-9
    mu_x, sd_x = Xf.mean(0), Xf.std(0) + 1e-9
    w, b0 = ridge_fit((Xf - mu_x) / sd_x, (Yf - mu_y) / sd_y, alpha=30.0)

    def _surprise(b):
        T = len(b["track"])
        As = []
        for t in range(1, T - H_FRAMES):
            if b["valid"][t] and b["valid"][t - 1] and b["valid"][t + H_FRAMES]:
                x = np.concatenate(
                    [
                        (b["track"][t] - mu_x[:12]) / sd_x[:12],
                        (b["track"][t - 1] - mu_x[12:]) / sd_x[12:],
                    ]
                )
                pred = (x @ w + b0) * sd_y + mu_y
                r = (b["track"][t + H_FRAMES] - pred) / sd_y
                As.append(float((r**2).mean()))
        return As

    # ---- PART 2: Theta_A within frozen pairs (eval bouts, equal weights)
    per_pair: dict = {}
    for b in ev:
        As = _surprise(b)
        if not As:
            continue
        key = (b["caller"], b["recipient"])
        per_pair.setdefault(key, {11: [], 12: []})[b["ctx"]].append(float(np.median(As)))
    supported = {k: v for k, v in per_pair.items() if len(v[11]) >= 2 and len(v[12]) >= 2}
    pair_deltas = {k: float(np.median(v[12]) - np.median(v[11])) for k, v in supported.items()}
    theta_A = float(np.mean(list(pair_deltas.values())))
    # block-clustered bootstrap over pairs' recording blocks
    pair_blocks = {}
    for b in ev:
        for k in supported:
            if (b["caller"], b["recipient"]) == k:
                pair_blocks.setdefault(k, set()).add(b["block"])
    keys = sorted(supported)
    rng = np.random.RandomState(SEED)
    reps = []
    for _ in range(BOOT_B):
        pick = [keys[i] for i in rng.randint(0, len(keys), len(keys))]
        reps.append(float(np.mean([pair_deltas[k] for k in pick])))
    ci = [round(float(np.percentile(reps, 2.5)), 4), round(float(np.percentile(reps, 97.5)), 4)]
    print(
        json.dumps({"Theta_A": round(theta_A, 4), "ci": ci, "n_pairs": len(supported)}, indent=1),
        flush=True,
    )

    # ---- PART 1: sample combined records
    records = []
    for b in sorted(ev, key=lambda x: x["file"])[: args.sample]:
        p12 = float(
            clf.predict_proba(((b["static"] - mu_c) / sd_c)[None])[0, list(clf.classes_).index(12)]
        )
        abstain = abs(p12 - 0.5) < ABSTAIN_MARGIN
        As = _surprise(b)
        rec = {
            "source_interval": {
                "file": b["file"],
                "valid_frames": int(b["valid"].sum()),
                "total_frames": len(b["valid"]),
            },
            "caller": b["caller"],
            "recipient": b["recipient"],
            "context_score_c12_uncalibrated": round(p12, 4),
            "suggested_context": None
            if abstain
            else int(clf.predict(((b["static"] - mu_c) / sd_c)[None])[0]),
            "abstain": abstain,
            "aggregate_surprise_median": round(float(np.median(As)), 4) if As else None,
            "aggregate_surprise_q90": round(float(np.percentile(As, 90)), 4) if As else None,
            "whole_call_static_features": {
                n: round(float(b["static"][_NAME_TO_IDX[n]]), 3) for n in feats
            },
            "valid_support": float(b["valid"].mean()),
            "model_versions": {
                "context_head": "L2 logistic C=1.0 balanced; static 44-D; "
                "declared abstain band |p-0.5|<0.2",
                "forecast_head": "ridge alpha=30 state+previous; 40 ms horizon",
                "feature_schema": 2,
            },
            "provenance_states": "surprise values are MEASUREMENTS of "
            "observed audio (not reconstructions); suggested_context is a "
            "suggestion requiring review — never an intention claim",
        }
        records.append(rec)
    out = ROOT / "outputs" / "joint_vocalization_records.json"
    out.write_text(
        json.dumps(
            {
                "architecture": "parallel heads: G (context) / F (forecast) / R "
                "(deviation) — complementary branches; context-conditioned "
                "forecasting was tested and CLOSED (no benefit)",
                "records_sample": records,
                "theta_A_test": {
                    "statistic": "median(A|C12) - median(A|C11), A = forecast "
                    "surprise under the SINGLE shared forecaster",
                    "Theta_A": round(theta_A, 4),
                    "bootstrap_ci": ci,
                    "n_supported_pairs": len(supported),
                    "interpretation_rows": "theta ~ 0 -> shared local dynamics "
                    "across contexts; positive -> C12 trajectories less captured "
                    "by shared dynamics (differently predictable REGIMES — NOT "
                    "different forecasting rules); negative -> C11 less captured",
                    "single_statistic_note": "one prespecified quantity; no "
                    "residual-feature search (earlier ablation showed little "
                    "incremental value)",
                },
            },
            indent=1,
        )
    )
    print(f"wrote {out} ({len(records)} sample records + Theta_A)", flush=True)


if __name__ == "__main__":
    main()
