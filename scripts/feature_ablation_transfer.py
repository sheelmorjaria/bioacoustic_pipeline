#!/usr/bin/env python3
"""Roadmap items 4+5: feature-ablation mechanics and cross-caller transfer.

(4) FEATURE ABLATION (prespecified groups of the 12 supported features):
    spectral-shape: centroid, spread, rolloff, flatness, entropy, dom_freq
    amplitude:      rms, peak, crest
    temporal-change: zcr, flux, dyn_cv
  For each group: remove its columns from the history input (state+prev+
  older lags), refit, and report Delta_ablation = L(ablated) - L(full)
  at h=40 ms on identical eval positions. Positive = the group carries
  temporal predictive signal. (Removal, not zeroing: the model never sees
  the dimension.)

(5) CROSS-CALLER TRANSFER: forecaster trained on a subset of callers,
  evaluated on HELD-OUT CALLERS with coefficients frozen (no refit, no
  scaler refit). Reported alongside the within-caller eval of the same
  model. Delta_transfer = L(held-out callers) - L(within), block
  clustered. This is model transfer (not procedure replication): a
  negative/unresolved result does not impugn the procedure-replication
  results already established across corpora.

Usage:
    python scripts/feature_ablation_transfer.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

H_FRAMES = 4
LAGS = (4, 8, 12, 20)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
BOOT_B = 1000
SEED = 0
TRAIN_FRAC = 0.7
VAL_FRAC = 0.25
N_HELDOUT_CALLERS = 5

GROUPS = {
    "spectral-shape": ("centroid", "spread", "rolloff", "flatness", "entropy", "dom_freq"),
    "amplitude": ("rms", "peak", "crest"),
    "temporal-change": ("zcr", "flux", "dyn_cv"),
}


def main() -> None:
    import numpy as np

    from acoustic_dynamics_report import ridge_fit, ridge_predict
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
    feats = list(supported_features())
    fidx = [_NAME_TO_IDX[n] for n in feats]

    bouts = []
    seen = set()
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
            f = frames[offsets[i] : offsets[i + 1]][:, fidx].astype(np.float64)
            valid = ~np.all(f == 0, axis=1)
            if valid.sum() >= 40:
                bouts.append(
                    {
                        "file": nm,
                        "block": filename_block_group(nm),
                        "caller": a.emitter,
                        "z": f,
                        "valid": valid,
                    }
                )
    assert_feature_contract(bouts[0]["z"], context="ablation/transfer fitting")

    # positions with full 6-block history
    def _collect_hist(selected):
        X, Y, BLK, CALL = [], [], [], []
        for b in selected:
            z, v = b["z"], b["valid"]
            T = len(z)
            for t in range(max(LAGS) + 1, T - H_FRAMES):
                if v[t] and v[t - 1] and v[t + H_FRAMES] and all(v[t - k] for k in LAGS):
                    hist = [z[t], z[t - 1]] + [z[t - k] for k in LAGS]
                    X.append(np.concatenate(hist))
                    Y.append(z[t + H_FRAMES])
                    BLK.append(b["block"])
                    CALL.append(b["caller"])
        return np.stack(X), np.stack(Y), np.array(BLK), np.array(CALL)

    X, Y, BLK, CALL = _collect_hist(bouts)
    print(f"positions: {len(X)}", flush=True)

    blocks = sorted(set(BLK.tolist()))
    fit_blocks = {
        b
        for b in blocks
        if int(hashlib.sha1(b.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    rng = np.random.RandomState(SEED)
    fit_list = sorted(fit_blocks)
    rng.shuffle(fit_list)
    val_blocks = set(fit_list[: max(1, int(len(fit_list) * VAL_FRAC))])
    fit_only = set(fit_list) - val_blocks
    is_fit = np.isin(BLK, sorted(fit_only))
    is_val = np.isin(BLK, sorted(val_blocks))
    is_ev = ~is_fit & ~is_val

    mu_x, sd_x = X[is_fit].mean(0), X[is_fit].std(0) + 1e-9
    mu_y, sd_y = Y[is_fit].mean(0), Y[is_fit].std(0) + 1e-9
    Xn, Yn = (X - mu_x) / sd_x, (Y - mu_y) / sd_y

    def _loss(r):
        return float(np.sqrt((r**2).mean(0)).mean())

    def _fit_eval(cols):
        A_tr = Xn[is_fit][:, cols]
        A_ev = Xn[is_ev][:, cols]
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(A_tr, Yn[is_fit], alpha)
            lv = _loss(ridge_predict(w, b, Xn[is_val][:, cols]) - Yn[is_val])
            if lv < best[1]:
                best = (alpha, lv)
        w, b = ridge_fit(A_tr, Yn[is_fit], best[0])
        r = ridge_predict(w, b, A_ev) - Yn[is_ev]
        return _loss(r), r, best[0]

    all_cols = list(range(X.shape[1]))
    L_full, r_full, a_full = _fit_eval(all_cols)

    # (4) ablation
    ablation = {}
    for gname, gfeats in GROUPS.items():
        drop_feat_idx = [feats.index(f) for f in gfeats]
        # history block b covers features 12*b .. 12*b+11
        drop_cols = [12 * b + j for b in range(6) for j in drop_feat_idx]
        keep = [c for c in all_cols if c not in drop_cols]
        L_ab, r_ab, a_ab = _fit_eval(keep)
        ev_pos = np.where(is_ev)[0]
        by = {}
        for j, i in enumerate(ev_pos):  # compressed eval indices
            by.setdefault(BLK[i], []).append(j)
        keys = sorted(by)
        brng = np.random.RandomState(SEED)
        reps = []
        for _ in range(BOOT_B):
            pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
            idx = np.concatenate([by[g] for g in pick])
            reps.append(_loss(r_ab[idx]) - _loss(r_full[idx]))
        ablation[gname] = {
            "delta_ablation": round(L_ab - L_full, 4),
            "ci95": [
                round(float(np.percentile(reps, 2.5)), 4),
                round(float(np.percentile(reps, 97.5)), 4),
            ],
            "L_ablated": round(L_ab, 4),
            "L_full": round(L_full, 4),
            "n_features_removed": len(gfeats),
        }
        print(f"ablate {gname}: +{L_ab - L_full:.4f} {ablation[gname]['ci95']}", flush=True)

    # (5) cross-caller transfer: same model, held-out CALLERS
    callers = sorted(set(CALL.tolist()))
    rng.shuffle(callers)
    held = set(callers[:N_HELDOUT_CALLERS])
    tr_mask = is_fit & ~np.isin(CALL, sorted(held))
    ev_within = is_ev & ~np.isin(CALL, sorted(held))
    ev_held = np.isin(CALL, sorted(held))  # any block: callers never seen
    mu2, sd2 = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-9
    mu_y2, sd_y2 = Y[tr_mask].mean(0), Y[tr_mask].std(0) + 1e-9
    Xn2 = (X - mu2) / sd2
    Yn2 = (Y - mu_y2) / sd_y2
    best = (None, np.inf)
    for alpha in ALPHAS:
        w, b = ridge_fit(Xn2[tr_mask], Yn2[tr_mask], alpha)
        lv = _loss(
            ridge_predict(w, b, Xn2[is_val & ~np.isin(CALL, sorted(held))])
            - Yn2[is_val & ~np.isin(CALL, sorted(held))]
        )
        if lv < best[1]:
            best = (alpha, lv)
    w, b = ridge_fit(Xn2[tr_mask], Yn2[tr_mask], best[0])
    r_within = ridge_predict(w, b, Xn2[ev_within]) - Yn2[ev_within]
    r_held = ridge_predict(w, b, Xn2[ev_held]) - Yn2[ev_held]
    delta_transfer = _loss(r_held) - _loss(r_within)
    ev_h = np.where(ev_held)[0]
    byb = {}
    for j, i in enumerate(ev_h):
        byb.setdefault(BLK[i], []).append(j)
    keys = sorted(byb)
    reps = []
    for _ in range(BOOT_B):
        pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
        idx = np.concatenate([byb[g] for g in pick])
        widx = brng.randint(0, len(r_within), len(idx))
        reps.append(_loss(r_held[idx]) - _loss(r_within[widx]))
    transfer = {
        "heldout_callers": sorted(held),
        "L_within": round(_loss(r_within), 4),
        "L_heldout_callers": round(_loss(r_held), 4),
        "delta_transfer": round(delta_transfer, 4),
        "note": "model transfer (frozen coefficients + scaler): a negative/"
        "unresolved result does not impugn the procedure-"
        "replication results (which refit per corpus)",
        "n_within": int(ev_within.sum()),
        "n_heldout": int(ev_held.sum()),
    }

    report = {
        "study": "roadmap items 4+5: feature-ablation mechanics + "
        "cross-caller model transfer (h=40 ms)",
        "ablation": ablation,
        "L_full_reference": round(L_full, 4),
        "transfer": transfer,
        "framing": "which dimensions carry temporal predictive signal; "
        "whether one caller-fitted forecaster transfers — predictive "
        "measurements, no meaning claims",
    }
    out = ROOT / "outputs" / "feature_ablation_transfer.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(transfer, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
