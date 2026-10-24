#!/usr/bin/env python3
"""Transfer-branch tightening: clustered CIs + OOD diagnostic + matched-size sensitivity.

(1) RECORDING-GROUP-RESAMPLED CIs on Delta_transfer per pair (from frozen
    predictions; no retraining).
(2) OOD / COORDINATE-SHIFT DIAGNOSTIC: per-feature source->target mean
    shift and scale ratio (in source standardised units), covariance
    mismatch (Frobenius of correlation difference), fraction of target
    states outside the source k=10 NN common support, and target
    per-position transfer error regressed on the per-position OOD score
    (mean standardised distance to the source k-NN). Asks: does transfer
    error increase with acoustic-coordinate OOD?
(3) MATCHED-TRAINING-SIZE SENSITIVITY (diagnostic only, NOT a primary
    endpoint): downsample the bat fit set to the target's fit size,
    refit Ridge under the identical policy, repeat bat->{chiffchaff,
    littleowl} transfer. Answers whether the frozen-beats-refit advantage
    survives rough control for source training volume.

Wording corrections applied throughout: "consistent with coordinate
distribution shift, but this study does not isolate the cause";
"the bat model's larger fitting set is ONE PLAUSIBLE CONTRIBUTER".

Usage:
    python scripts/transfer_tightening.py

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

from cross_corpus_transfer import (
    ALPHAS,
    H_FRAMES,
    LAGS,
    PANEL,
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    _load_avian,
    _load_bat,
)
from scipy.spatial import cKDTree

BOOT_B = 1000
K_NN = 10


def _positions(units):
    import numpy as np

    X6, Y, GRP = [], [], []
    for b in units:
        z, v = b["z"], b["valid"]
        T = len(z)
        for t in range(max(LAGS) + 1, T - H_FRAMES):
            if v[t] and v[t - 1] and v[t + H_FRAMES] and all(v[t - k] for k in LAGS):
                X6.append(np.concatenate([z[t], z[t - 1]] + [z[t - k] for k in LAGS]))
                Y.append(z[t + H_FRAMES])
                GRP.append(b["group"])
    return np.stack(X6), np.stack(Y), np.array(GRP)


def _split(groups):
    import numpy as np

    gs = sorted(set(groups.tolist()))
    fit = {
        g for g in gs if int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    rng = np.random.RandomState(SEED)
    fl = sorted(fit)
    rng.shuffle(fl)
    val = set(fl[: max(1, int(len(fl) * VAL_FRAC))])
    fit_only = set(fl) - val
    return (
        np.isin(groups, sorted(fit_only)),
        np.isin(groups, sorted(val)),
        ~np.isin(groups, sorted(fit_only | val)),
    )


def _loss(r):
    import numpy as np

    return float(np.sqrt((r**2).mean(0)).mean())


def _fit(X, Y, is_fit, is_val):
    import numpy as np

    from acoustic_dynamics_report import ridge_fit, ridge_predict

    mu_x, sd_x = X[is_fit].mean(0), X[is_fit].std(0) + 1e-9
    mu_y, sd_y = Y[is_fit].mean(0), Y[is_fit].std(0) + 1e-9
    Xn = (X - mu_x) / sd_x
    Yn = (Y - mu_y) / sd_y
    best = (None, np.inf)
    for alpha in ALPHAS:
        w, b = ridge_fit(Xn[is_fit], Yn[is_fit], alpha)
        lv = _loss(ridge_predict(w, b, Xn[is_val]) - Yn[is_val])
        if lv < best[1]:
            best = (alpha, lv)
    w, b = ridge_fit(Xn[is_fit], Yn[is_fit], best[0])
    return {
        "mu_x": mu_x,
        "sd_x": sd_x,
        "mu_y": mu_y,
        "sd_y": sd_y,
        "w": w,
        "b": b,
        "alpha": best[0],
    }


def _apply_resid(model, X, Y):
    from acoustic_dynamics_report import ridge_predict

    Xn = (X - model["mu_x"]) / model["sd_x"]
    return ridge_predict(model["w"], model["b"], Xn) - (Y - model["mu_y"]) / model["sd_y"]


def _boot_ci(resid_frozen, resid_refit, groups, b=BOOT_B):
    import numpy as np

    by = {}
    for j in range(len(resid_frozen)):
        by.setdefault(groups[j], []).append(j)
    keys = sorted(by)
    brng = np.random.RandomState(SEED)

    def _l(r):
        return float(np.sqrt((r**2).mean(0)).mean())

    obs = _l(resid_frozen) - _l(resid_refit)
    reps = []
    for _ in range(b):
        pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
        idx = np.concatenate([by[g] for g in pick])
        reps.append(_l(resid_frozen[idx]) - _l(resid_refit[idx]))
    return (
        round(obs, 4),
        [round(float(np.percentile(reps, 2.5)), 4), round(float(np.percentile(reps, 97.5)), 4)],
    )


def main() -> None:
    import numpy as np

    sources = {}
    for src in ("bat", "chiffchaff"):
        units = _load_bat() if src == "bat" else _load_avian(src)
        X, Y, G = _positions(units)
        is_f, is_v, _ = _split(G)
        sources[src] = (units, _fit(X, Y, is_f, is_v), int(is_f.sum()))
        print(f"source {src}: {int(is_f.sum())} fit positions", flush=True)

    results = {}
    for src, tgt in PANEL:
        units = _load_avian(tgt)
        X, Y, G = _positions(units)
        is_f, is_v, is_e = _split(G)
        tgt_model = _fit(X, Y, is_f, is_v)
        src_units, src_model, n_src_fit = sources[src]
        resid_frozen = _apply_resid(src_model, X[is_e], Y[is_e])
        resid_refit = _apply_resid(tgt_model, X[is_e], Y[is_e])
        delta, ci = _boot_ci(resid_frozen, resid_refit, G[is_e])

        # OOD diagnostic (state space = first 12 cols = current state)
        Xs_src, Ys_src, Gs_src = _positions(src_units)
        isf, _, _ = _split(Gs_src)
        Ssrc = Xs_src[isf][:, :12]
        Ssrc_n = (Ssrc - src_model["mu_x"][:12]) / src_model["sd_x"][:12]
        Stgt_n = (X[is_e][:, :12] - src_model["mu_x"][:12]) / src_model["sd_x"][:12]
        tree = cKDTree(Ssrc_n)
        dist, _ = tree.query(Stgt_n, k=K_NN)
        ood = dist.mean(axis=1)
        err = np.sqrt((resid_frozen**2).mean(axis=1))
        # quartile contrast: error in top-OOD quartile vs bottom
        q75 = float(np.percentile(ood, 75))
        hi = err[ood >= q75].mean()
        lo = err[ood < q75].mean()
        # per-feature shift stats (source-standardised)
        mean_shift = Stgt_n.mean(axis=0)
        scale_ratio = Stgt_n.std(axis=0) / (Ssrc_n.std(axis=0) + 1e-12)
        # outside-support fraction: dist to nearest source state > source 95th pct NN dist
        d1, _ = tree.query(Stgt_n, k=1)
        ref_d, _ = tree.query(Ssrc_n[::37][:4000], k=2)
        thr = float(np.percentile(ref_d[:, 1], 95))
        outside = float((d1 > thr).mean())
        corr_src = np.corrcoef(Ssrc_n[::13].T)
        corr_tgt = np.corrcoef(Stgt_n[::37].T)
        cov_mismatch = float(np.linalg.norm(corr_tgt - corr_src))

        results[f"{src}->{tgt}"] = {
            "Delta_transfer": delta,
            "ci95": ci,
            "n_eval": int(is_e.sum()),
            "ood": {
                "frac_outside_source_support": round(outside, 3),
                "cov_corr_mismatch_frobenius": round(cov_mismatch, 3),
                "mean_abs_feature_shift": round(float(np.abs(mean_shift).mean()), 3),
                "max_abs_feature_shift": round(float(np.abs(mean_shift).max()), 3),
                "mean_scale_ratio": round(float(scale_ratio.mean()), 3),
                "err_topOOD_quartile": round(float(hi), 4),
                "err_bottomOOD_quartile": round(float(lo), 4),
                "err_ratio": round(float(hi / max(lo, 1e-12)), 3),
            },
        }
        print(
            f"{src}->{tgt}: delta {delta:+.4f} CI {ci} | outside-support "
            f"{outside:.1%} | err ratio top/bottom OOD {hi / max(lo, 1e-12):.2f}",
            flush=True,
        )

    # matched-training-size sensitivity (diagnostic)
    sensitivity = {}
    for tgt in ("chiffchaff", "littleowl"):
        units = _load_avian(tgt)
        X, Y, G = _positions(units)
        is_f, is_v, is_e = _split(G)
        tgt_model = _fit(X, Y, is_f, is_v)
        src_units, _, n_src = sources["bat"]
        Xs, Ys, Gs = _positions(src_units)
        isf, ivf, _ = _split(Gs)
        rng = np.random.RandomState(SEED)
        idx = rng.choice(int(isf.sum()), size=min(int(isf.sum()), int(is_f.sum())), replace=False)
        Xs_sub, Ys_sub = Xs[isf][idx], Ys[isf][idx]
        sub_model = _fit(Xs_sub, Ys_sub, np.ones(len(Xs_sub), bool), np.ones(len(Xs_sub), bool))
        resid_f = _apply_resid(sub_model, X[is_e], Y[is_e])
        resid_r = _apply_resid(tgt_model, X[is_e], Y[is_e])
        d, ci = _boot_ci(resid_f, resid_r, G[is_e])
        sensitivity[f"bat({int(is_f.sum())})-matched->{tgt}"] = {
            "n_bat_subsample": int(len(Xs_sub)),
            "n_target_fit": int(is_f.sum()),
            "alpha_sub": sub_model["alpha"],
            "Delta_transfer": d,
            "ci95": ci,
            "note": "diagnostic sensitivity only — NOT a primary endpoint",
        }
        print(f"matched bat({len(Xs_sub)})->{tgt}: delta {d:+.4f} CI {ci}", flush=True)

    report = {
        "study": "transfer-branch tightening: clustered CIs + OOD diagnostic "
        "+ matched-size sensitivity",
        "ci_method": "recording-group-resampled paired bootstrap B=1000 "
        "(from frozen predictions; no retraining)",
        "results": results,
        "matched_size_sensitivity": sensitivity,
        "wording_corrections": [
            "pipit failure is CONSISTENT WITH coordinate distribution shift; "
            "this study does not isolate the cause (alpha=1000 is "
            "suggestive, not diagnostic: collinearity, low effective signal, "
            "sample size, noisy targets and covariance differences can "
            "produce the same pattern)",
            "the bat model's larger fitting set is ONE PLAUSIBLE "
            "CONTRIBUTOR to its superior transfer on chiffchaff/little owl; "
            "a matched-data-size or learning-curve comparison is required "
            "to establish it (provided here as the sensitivity analysis)",
        ],
    }
    out = ROOT / "outputs" / "transfer_tightening.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
