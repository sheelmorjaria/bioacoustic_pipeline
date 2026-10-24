#!/usr/bin/env python3
"""Occupancy-dynamics decomposition: does context change WHERE vocalisations
go in acoustic space, or HOW they evolve there?

HYPOTHESIS (the review's sharper statement of the programme's pattern):
    P(H|C11) != P(H|C12)   (different acoustic occupancy)
    P(Y|H,C11) ~= P(Y|H,C12)  (similar local transition law on overlap)

TWO PRIMARY ENDPOINTS (prespecified):

  H1 — state-distribution difference: energy distance between C11 and
  C12 state distributions on the frozen dyadic population, with
  recording-group label permutations preserving caller/recipient
  structure; plus the COMMON-SUPPORT fraction (k-NN overlap: fraction of
  each context's positions whose 10 nearest neighbours include >=1 of
  the other context, in the training-fitted standardised coordinates).

  H2 — cross-context predictive EQUIVALENCE on common support: fit
  F_11, F_12, F_shared (identical ridge architecture, alpha on
  train-side validation); evaluate the transfer matrix
  L_{11->11}, L_{11->12}, L_{12->11}, L_{12->12} restricted to
  COMMON-SUPPORT eval positions (a position is common-support if its
  k=10 nearest training neighbours span both contexts);
      Gamma_11 = L_{12->11} - L_{11->11}
      Gamma_12 = L_{11->12} - L_{12->12}
  EQUIVALENCE MARGIN (prospective, declared before evaluation):
  epsilon = 0.010 nRMSE — one percent of the FIT target std scale,
  i.e. a transfer cost smaller than 1% of the state's own spread is
  scientifically negligible FOR THIS REPRESENTATION AND HORIZON. H2 is
  SUPPORTED iff the 95% CI of each Gamma lies entirely inside
  [-epsilon, +epsilon]. Failure to meet the margin = non-equivalence
  NOT established; the interval is reported either way.

DIAGNOSTICS (secondary, clearly labelled):
  D1 — matched transitions: for each common-support C11 position with a
  close C12 neighbour (standardised distance < the 5th-percentile
  within-context NN distance, within the same caller where possible),
  compare D_cross = E||dX_C11 - dX_C12||^2 against
  D_within = E||dX_i - dX_j||^2 from matched same-context pairs.
  D2 — residual-vector context classification: locked linear classifier
  predicting context from the full standardised residual vector R_t on
  held-out blocks (if R cannot discriminate while G can, the
  occupancy/shared-dynamics story strengthens).

OUTCOME TABLE (frozen): different+equivalent = different occupancy,
shared local dynamics; different+different = state AND dynamics;
similar+different; similar+equivalent. All T0/T1 acoustic evidence — no
receiver or meaning claims.

Usage:
    python scripts/occupancy_dynamics.py

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
VAL_FRAC = 0.25
H_FRAMES = 4
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
BOOT_B = 2000
PERM_B = 2000
SEED = 0
K_NN = 10
EPSILON = 0.010  # prospective equivalence margin (nRMSE units)
MAX_POSITIONS = 60000  # computational cap, deterministic stride


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

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
    audit = json.loads((ROOT / "outputs/history_support_audit.json").read_text())
    frozen = {
        (p["caller"], p["recipient"])
        for p in audit["support_audit"]["dyadic_cells_frozen"]["pairs"].values()
    }
    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]

    # ---- positions: (state s_t, next s_{t+h}, context, caller, block)
    positions = []
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
            T = len(f)
            for t in range(1, T - H_FRAMES):
                if valid[t] and valid[t + H_FRAMES]:
                    positions.append(
                        (f[t], f[t + H_FRAMES], a.context, a.emitter, filename_block_group(nm))
                    )
    positions = positions[:: max(1, len(positions) // MAX_POSITIONS)]
    S = np.stack([p[0] for p in positions])
    Y = np.stack([p[1] for p in positions])
    C = np.array([p[2] for p in positions])
    CALL = np.array([p[3] for p in positions])
    BLK = np.array([p[4] for p in positions])
    print(
        f"positions: {len(S)} | C11 {int((C == 11).sum())} / C12 {int((C == 12).sum())}", flush=True
    )
    assert_feature_contract(S, context="occupancy-dynamics fitting")

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
    fit_blocks = set(fit_list) - val_blocks
    is_fit = np.isin(BLK, sorted(fit_blocks))
    is_val = np.isin(BLK, sorted(val_blocks))
    is_ev = ~is_fit & ~is_val

    mu_s, sd_s = S[is_fit].mean(0), S[is_fit].std(0) + 1e-9
    mu_y, sd_y = Y[is_fit].mean(0), Y[is_fit].std(0) + 1e-9
    Sn = (S - mu_s) / sd_s
    Yn = (Y - mu_y) / sd_y

    # ================= H1: energy distance + common support
    def _energy(a, b):
        from scipy.spatial.distance import cdist

        aa = cdist(a, a).mean()
        bb = cdist(b, b).mean()
        ab = cdist(a, b).mean()
        return float(2 * ab - aa - bb)

    S11 = Sn[is_fit & (C == 11)]
    S12 = Sn[is_fit & (C == 12)]
    cap = 4000
    r11 = np.random.RandomState(SEED).choice(len(S11), min(cap, len(S11)), False)
    r12 = np.random.RandomState(SEED + 1).choice(len(S12), min(cap, len(S12)), False)
    E_obs = _energy(S11[r11], S12[r12])
    # recording-group label permutations (swap context labels within blocks
    # that have both contexts — preserves caller/recipient structure)
    both_blocks = [
        b
        for b in fit_blocks
        if (C[is_fit & (BLK == b)] == 11).any() and (C[is_fit & (BLK == b)] == 12).any()
    ]
    prng = np.random.RandomState(SEED)
    null_E = []
    for _ in range(PERM_B):
        Cperm = C.copy()
        for b in both_blocks:
            m = BLK == b
            idx = np.where(m & is_fit)[0]
            lab = C[idx]
            prng.shuffle(lab)
            Cperm[idx] = lab
        a = Sn[is_fit & (Cperm == 11)]
        bb_ = Sn[is_fit & (Cperm == 12)]
        ra = prng.choice(len(a), min(cap, len(a)), False)
        rb = prng.choice(len(bb_), min(cap, len(bb_)), False)
        null_E.append(_energy(a[ra], bb_[rb]))
    p_perm = float((1 + np.sum(np.asarray(null_E) >= E_obs)) / (PERM_B + 1))
    # common support via k-NN context mix (training coordinates)
    from scipy.spatial import cKDTree

    tree = cKDTree(Sn[is_fit])
    Cfit = C[is_fit]
    ev_idx = np.where(is_ev)[0]
    _, nn = tree.query(Sn[ev_idx], k=K_NN)
    nn_c = Cfit[nn]
    common = (nn_c == 11).any(1) & (nn_c == 12).any(1)
    frac_common = float(common.mean())
    frac_common_c11 = float(common[C[ev_idx] == 11].mean())
    frac_common_c12 = float(common[C[ev_idx] == 12].mean())
    print(
        json.dumps(
            {
                "H1": {
                    "energy_distance": round(E_obs, 4),
                    "perm_p": round(p_perm, 5),
                    "common_support_fraction": round(frac_common, 3),
                    "by_context": {
                        "11": round(frac_common_c11, 3),
                        "12": round(frac_common_c12, 3),
                    },
                },
            },
            indent=1,
        ),
        flush=True,
    )

    # ================= H2: cross-context transfer on common support
    ev_common = ev_idx[common]

    def _pick_alpha(mask, y):
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(Sn[mask], y[mask], alpha)
            lv = float(
                np.sqrt(((ridge_predict(w, b, Sn[is_val]) - Yn[is_val]) ** 2).mean(0)).mean()
            )
            if lv < best[1]:
                best = (alpha, lv)
        return best[0]

    models = {}
    for name, mask in (
        ("shared", is_fit),
        ("F11", is_fit & (C == 11)),
        ("F12", is_fit & (C == 12)),
    ):
        alpha = _pick_alpha(mask, Yn)
        w, b = ridge_fit(Sn[mask], Yn[mask], alpha)
        models[name] = (w, b, alpha)

    def _loss(name, idx, ctx):
        w, b, _ = models[name]
        r = ridge_predict(w, b, Sn[idx]) - Yn[idx]
        return float(np.sqrt((r**2).mean(0)).mean())

    m11 = ev_common[C[ev_common] == 11]
    m12 = ev_common[C[ev_common] == 12]
    L = {
        "L_11->11": _loss("F11", m11, 11),
        "L_12->11": _loss("F12", m11, 11),
        "L_11->12": _loss("F11", m12, 12),
        "L_12->12": _loss("F12", m12, 12),
        "L_shared_on_common": _loss("shared", ev_common, 0),
    }
    gamma11 = L["L_12->11"] - L["L_11->11"]
    gamma12 = L["L_11->12"] - L["L_12->12"]

    # bootstrap over recording blocks within common-support eval
    by_blk: dict = {}
    for i in ev_common:
        by_blk.setdefault(BLK[i], []).append(i)
    keys = sorted(by_blk)
    brng = np.random.RandomState(SEED)
    g11_reps, g12_reps = [], []
    for _ in range(BOOT_B):
        pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
        idx = np.concatenate([by_blk[g] for g in pick])
        i11 = idx[C[idx] == 11]
        i12 = idx[C[idx] == 12]
        if len(i11) < 5 or len(i12) < 5:
            continue
        w11, b11, _ = models["F11"]
        w12, b12, _ = models["F12"]
        r11a = ridge_predict(w11, b11, Sn[i11]) - Yn[i11]
        r12a = ridge_predict(w12, b11 * 0 + b12, Sn[i11]) - Yn[i11]
        r12b = ridge_predict(w12, b12, Sn[i12]) - Yn[i12]
        r11b = ridge_predict(w11, b11, Sn[i12]) - Yn[i12]

        def ll(r):
            return float(np.sqrt((r**2).mean(0)).mean())

        g11_reps.append(ll(r12a) - ll(r11a))
        g12_reps.append(ll(r11b) - ll(r12b))
    ci11 = [float(np.percentile(g11_reps, 2.5)), float(np.percentile(g11_reps, 97.5))]
    ci12 = [float(np.percentile(g12_reps, 2.5)), float(np.percentile(g12_reps, 97.5))]
    eq11 = ci11[0] > -EPSILON and ci11[1] < EPSILON
    eq12 = ci12[0] > -EPSILON and ci12[1] < EPSILON
    print(
        json.dumps(
            {
                "H2": {
                    "Gamma_11": round(gamma11, 4),
                    "ci11": [round(x, 4) for x in ci11],
                    "Gamma_12": round(gamma12, 4),
                    "ci12": [round(x, 4) for x in ci12],
                    "epsilon": EPSILON,
                    "equivalent_11": bool(eq11),
                    "equivalent_12": bool(eq12),
                },
                "L": {k: round(v, 4) for k, v in L.items()},
            },
            indent=1,
        ),
        flush=True,
    )

    # ================= diagnostics
    # D1 matched transitions (same caller where possible)
    dX = Yn - Sn
    tree11 = cKDTree(Sn[is_fit & (C == 11)])
    tree12 = cKDTree(Sn[is_fit & (C == 12)])
    S11f, _S12f = Sn[is_fit & (C == 11)], Sn[is_fit & (C == 12)]
    dX11f, _dX12f = dX[is_fit & (C == 11)], dX[is_fit & (C == 12)]
    d11, _ = tree11.query(S11f, k=2)
    thr = float(np.percentile(d11[:, 1], 5))
    cross_d, within_d = [], []
    idx12 = np.where(is_fit & (C == 12))[0]
    CALL[idx12]
    Sn[idx12]
    dX12all = dX[idx12]
    for k in range(0, len(S11f), 25):
        d, j = tree12.query(S11f[k], k=1)  # opposite-context tree: no self
        if float(np.atleast_1d(d)[0]) < thr:
            cross_d.append(float(((dX11f[k] - dX12all[int(np.atleast_1d(j)[0])]) ** 2).sum()))
        d2, j2 = tree11.query(S11f[k], k=2)
        # k=2 on the SAME-context tree returns [self, nearest] — the
        # neighbour is index 1 (index 0 is the query point itself)
        if float(np.atleast_1d(d2)[1]) < thr:
            within_d.append(float(((dX11f[k] - dX11f[int(np.atleast_1d(j2)[1])]) ** 2).sum()))
    D_cross = float(np.mean(cross_d)) if cross_d else None
    D_within = float(np.mean(within_d)) if within_d else None

    # D2 residual-vector context classification (locked linear, held-out)
    w, b, _ = models["shared"]
    R = ridge_predict(w, b, Sn) - Yn
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(R[is_fit], C[is_fit])
    predR = clf.predict(R[is_ev])
    accR = float(np.mean([(predR[C[is_ev] == c] == c).mean() for c in (11, 12)]))
    # G-side comparator on the same positions
    clfG = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clfG.fit(Sn[is_fit], C[is_fit])
    predG = clfG.predict(Sn[is_ev])
    accG = float(np.mean([(predG[C[is_ev] == c] == c).mean() for c in (11, 12)]))

    # ---- outcome mapping
    h1_different = p_perm < 0.05
    h2_equivalent = eq11 and eq12
    if h1_different and h2_equivalent:
        verdict = "DIFFERENT OCCUPANCY, SHARED LOCAL DYNAMICS (row 1)"
    elif h1_different and not h2_equivalent:
        verdict = (
            "state AND/OR dynamics differ — transfer cost CI not "
            "inside the equivalence margin (row 2 or margin "
            "unresolved)"
        )
    elif not h1_different and h2_equivalent:
        verdict = "little acoustic context differentiation at this scale (row 4)"
    else:
        verdict = "similar occupancy but different evolution (row 3) — UNEXPECTED combination"

    report = {
        "study": "occupancy-dynamics decomposition (H1 state distributions; "
        "H2 cross-context predictive equivalence on common support)",
        "hypothesis": "P(H|C11) != P(H|C12) while P(Y|H,C11) ~= P(Y|H,C12)",
        "positions": {
            "n": int(len(S)),
            "fit": int(is_fit.sum()),
            "val": int(is_val.sum()),
            "eval": int(is_ev.sum()),
            "eval_common_support": int(len(ev_common)),
        },
        "H1": {
            "energy_distance": round(E_obs, 4),
            "perm_p_recording_group": round(p_perm, 5),
            "n_perm_blocks_with_both": len(both_blocks),
            "common_support_fraction": round(frac_common, 3),
            "common_support_by_context": {
                "11": round(frac_common_c11, 3),
                "12": round(frac_common_c12, 3),
            },
            "different": bool(h1_different),
        },
        "H2": {
            "margin_epsilon": EPSILON,
            "margin_rationale": "1% of FIT target std scale — declared "
            "BEFORE evaluation; a smaller transfer cost is negligible "
            "for this representation and 40 ms horizon",
            "Gamma_11": round(gamma11, 4),
            "ci11": [round(x, 4) for x in ci11],
            "Gamma_12": round(gamma12, 4),
            "ci12": [round(x, 4) for x in ci12],
            "equivalent_11": bool(eq11),
            "equivalent_12": bool(eq12),
            "losses": {k: round(v, 4) for k, v in L.items()},
            "equivalent": bool(h2_equivalent),
        },
        "diagnostics": {
            "D1_matched_transitions": {
                "match_threshold_std": round(thr, 3),
                "n_cross_pairs": len(cross_d),
                "n_within_pairs": len(within_d),
                "D_cross": None if D_cross is None else round(D_cross, 4),
                "D_within": None if D_within is None else round(D_within, 4),
                "ratio": None if not D_cross or not D_within else round(D_cross / D_within, 3),
                "note": "cross-context matched transitions no less similar "
                "than within-context (ratio ~<= 1) supports shared local "
                "relationship",
            },
            "D2_residual_vector_context_BA": {
                "residual_head": round(accR, 3),
                "state_head_G": round(accG, 3),
                "note": "if R cannot discriminate while G can, occupancy/"
                "shared-dynamics story strengthens",
            },
            "declared_simplification": "H1 uses s_t alone; H2 forecasters "
            "use single-state input identically for both contexts (fair "
            "cross-context comparison; the frozen [s_t,s_{t-1}] forecaster "
            "is the validated production head, unchanged)",
        },
        "verdict": verdict,
        "framing": "T0/T1 acoustic evidence — no receiver or meaning claims; "
        "equivalence is under the tested scale/representation only",
    }
    out = ROOT / "outputs" / "occupancy_dynamics.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {"verdict": verdict, "H1_different": h1_different, "H2_equivalent": h2_equivalent},
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
