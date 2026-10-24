#!/usr/bin/env python3
"""Continuous-state decomposition: occupancy / drift / diffusion by context.

The review's three-question analog decomposition, operationalised on the
frozen bat dyadic population (common acoustic support throughout):

  p(x|C)    occupancy — WHERE states occur (measured already as H1 of the
            occupancy-dynamics study: different, 98% common support; reused
            here as the background condition, not recomputed)
  v_C(x)    drift/flow — do contexts MOVE differently from the same state?
            nearest-neighbour estimate of the expected 40 ms displacement
            field per context on common support; primary statistic
            D_v = E||v_11(x) - v_12(x)||^2 over common-support eval
            states, referenced against within-context field disagreement
            (split-half v_11^a vs v_11^b) so the cross-context difference
            is interpretable as a multiple of sampling noise
  Sigma_C(x) diffusion/variability — is one context's evolution more
            variable? local residual variance (displacement dispersion
            around each context's own field) per context, with
            block-clustered bootstrap on the variance RATIO

DIAGNOSTIC — trajectory geometry by context (descriptive): path length
per second, mean curvature, dwell (fraction of steps with speed below
the corpus q25), turning-angle circular mean — computed on full valid
spans, common-support unweighted. NO token claims.

FRAMING: an OPERATIONAL stochastic-dynamical description of acoustic
trajectories (dx = v_C(x) dt + B_C(x) dW as a model, not evidence the
production system obeys it). Continuous-state / analog-VALUED
computation — no hardware-analog claim.

Usage:
    python scripts/continuous_state_decomposition.py

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

H_FRAMES = 4
BOOT_B = 1000
SEED = 0
TRAIN_FRAC = 0.7
K_NN = 10
MAX_POSITIONS = 60000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    import numpy as np
    from scipy.spatial import cKDTree

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        contiguous_spans,
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

    positions = []
    spans_geo = []
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
            if (a.emitter, a.addressee) not in frozen:
                continue
            f = frames[offsets[i] : offsets[i + 1]][:, fidx].astype(np.float64)
            valid = ~np.all(f == 0, axis=1)
            blk = filename_block_group(nm)
            T = len(f)
            for t in range(1, T - H_FRAMES):
                if valid[t] and valid[t + H_FRAMES]:
                    positions.append((f[t], f[t + H_FRAMES] - f[t], a.context, blk))
            # full valid spans for geometry (>=30 frames)
            for sid in sorted(set(contiguous_spans(valid).tolist())):
                if sid < 0:
                    continue
                idx = np.where(contiguous_spans(valid) == sid)[0]
                if len(idx) >= 30:
                    spans_geo.append((f[idx], a.context))
    positions = positions[:: max(1, len(positions) // MAX_POSITIONS)]
    S = np.stack([p[0] for p in positions])
    DX = np.stack([p[1] for p in positions])
    C = np.array([p[2] for p in positions])
    BLK = np.array([p[3] for p in positions])
    assert_feature_contract(S, context="continuous decomposition fitting")
    print(f"positions: {len(S)} | geometry spans: {len(spans_geo)}", flush=True)

    blocks = sorted(set(BLK.tolist()))
    fit_blocks = {
        b
        for b in blocks
        if int(hashlib.sha1(b.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    is_fit = np.isin(BLK, sorted(fit_blocks))
    ev = ~is_fit
    mu, sd = S[is_fit].mean(0), S[is_fit].std(0) + 1e-9
    Sn = (S - mu) / sd
    DXn = DX / sd  # displacement in standardised units

    # common support on eval
    tree = cKDTree(Sn[is_fit])
    Cfit = C[is_fit]
    _, nn = tree.query(Sn[ev], k=K_NN)
    nn_c = Cfit[nn]
    common = (nn_c == 11).any(1) & (nn_c == 12).any(1)
    ev_c = np.where(ev)[0][common]
    print(f"common-support eval positions: {len(ev_c)}", flush=True)

    # ---- DRIFT: v_C(x) via K-NN expected displacement (train per context)
    fit11 = np.where(is_fit & (C == 11))[0]
    fit12 = np.where(is_fit & (C == 12))[0]
    t11 = cKDTree(Sn[fit11])
    t12 = cKDTree(Sn[fit12])
    DX11 = DXn[fit11]
    DX12 = DXn[fit12]

    def _field(tree, DXmat, queries, k=25):
        _, nn = tree.query(queries, k=k)
        return DXmat[nn].mean(axis=1)

    v11 = _field(t11, DX11, Sn[ev_c])
    v12 = _field(t12, DX12, Sn[ev_c])
    D_v = float(((v11 - v12) ** 2).sum(1).mean())
    # within-context split-half reference (sampling noise of the field)
    rng = np.random.RandomState(SEED)
    half = rng.permutation(len(fit11))
    a11, b11 = fit11[half[: len(half) // 2]], fit11[half[len(half) // 2 :]]
    half2 = rng.permutation(len(fit12))
    a12, b12 = fit12[half2[: len(half2) // 2]], fit12[half2[len(half2) // 2 :]]
    v11a = _field(cKDTree(Sn[a11]), DXn[a11], Sn[ev_c])
    v11b = _field(cKDTree(Sn[b11]), DXn[b11], Sn[ev_c])
    v12a = _field(cKDTree(Sn[a12]), DXn[a12], Sn[ev_c])
    v12b = _field(cKDTree(Sn[b12]), DXn[b12], Sn[ev_c])
    D_within = float((((v11a - v11b) ** 2).sum(1).mean() + ((v12a - v12b) ** 2).sum(1).mean()) / 2)
    # block bootstrap of the RATIO
    by_blk = {}
    for j, i in enumerate(ev_c):
        by_blk.setdefault(BLK[i], []).append(j)
    keys = sorted(by_blk)
    brng = np.random.RandomState(SEED)
    ratio_reps = []
    for _ in range(BOOT_B):
        pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
        idx = np.concatenate([by_blk[g] for g in pick])
        ratio_reps.append(float(((v11[idx] - v12[idx]) ** 2).sum(1).mean() / max(D_within, 1e-12)))
    ci_ratio = [float(np.percentile(ratio_reps, 2.5)), float(np.percentile(ratio_reps, 97.5))]

    # ---- DIFFUSION: local dispersion around each context's own field
    def _local_disp(tree, DXmat, queries, k=25):
        _, nn = tree.query(queries, k=k)
        disp = DXmat[nn] - DXmat[nn].mean(axis=1, keepdims=True)
        return (disp**2).sum(2).mean(axis=1)  # per-query dispersion

    m11 = C[ev_c] == 11
    m12 = C[ev_c] == 12
    disp11 = _local_disp(t11, DX11, Sn[ev_c][m11])
    disp12 = _local_disp(t12, DX12, Sn[ev_c][m12])
    ratio_diff = float(disp12.mean() / max(disp11.mean(), 1e-12))
    blk11 = BLK[ev_c][m11]
    blk12 = BLK[ev_c][m12]
    u11 = sorted(set(blk11.tolist()))
    u12 = sorted(set(blk12.tolist()))
    r_reps = []
    for _ in range(BOOT_B):
        p11 = [u11[i] for i in brng.randint(0, len(u11), len(u11))]
        p12 = [u12[i] for i in brng.randint(0, len(u12), len(u12))]
        d11p = np.concatenate([disp11[blk11 == g] for g in p11])
        d12p = np.concatenate([disp12[blk12 == g] for g in p12])
        r_reps.append(float(d12p.mean() / max(d11p.mean(), 1e-12)))
    ci_diff = [float(np.percentile(r_reps, 2.5)), float(np.percentile(r_reps, 97.5))]

    # ---- GEOMETRY diagnostics (descriptive)
    geo = {}
    for ctx in (11, 12):
        pl, _cur, dwell, turn = [], [], [], []
        for z, c in spans_geo:
            if c != ctx:
                continue
            dz = np.diff(z, axis=0) / sd
            speed = np.linalg.norm(dz, axis=1)
            pl.append(float(speed.mean()) * 100)  # path length / s
            if len(dz) >= 3:
                v1, v2 = dz[:-1], dz[1:]
                n1 = np.linalg.norm(v1, axis=1) + 1e-12
                n2 = np.linalg.norm(v2, axis=1) + 1e-12
                cosang = np.clip((v1 * v2).sum(1) / (n1 * n2), -1, 1)
                turn.append(float(np.arccos(cosang).mean()))
            allspeed = np.linalg.norm(np.diff(z / sd, axis=0), axis=1)
            q25 = float(
                np.percentile(
                    np.concatenate(
                        [
                            np.linalg.norm(np.diff(zz / sd, axis=0), axis=1)
                            for zz, cc in spans_geo
                            if cc == ctx
                        ]
                    ),
                    25,
                )
            )
            dwell.append(float((allspeed < q25).mean()))
        geo[str(ctx)] = {
            "path_length_per_s": round(float(np.median(pl)), 4),
            "mean_turning_angle_rad": round(float(np.median(turn)), 4),
            "dwell_fraction_below_corpus_q25": round(float(np.median(dwell)), 4),
            "n_spans": len(pl),
        }

    files_note = "frozen dyadic population; common-support evaluation"
    report = {
        "study": "continuous-state decomposition: occupancy / drift / "
        "diffusion by context (operational stochastic-dynamical model)",
        "framing": "dx = v_C(x) dt + B_C(x) dW is an OPERATIONAL model "
        "of acoustic trajectories, not evidence the production system "
        "obeys it; continuous-state (analog-valued) computation",
        "occupancy_background": "reused from the occupancy-dynamics study: "
        "different (energy 0.0167, perm p 0.001) with 98% common support",
        "drift": {
            "D_v_cross": round(D_v, 6),
            "D_v_within_split_half": round(D_within, 6),
            "ratio_cross_over_within": round(D_v / max(D_within, 1e-12), 3),
            "ratio_ci95": [round(x, 3) for x in ci_ratio],
            "interpretation": "ratio ~1 => context fields agree as much as "
            "split-half sampling of ONE context's field (shared drift); "
            "ratio >> 1 => contexts move differently from the same states",
            "n_eval": int(len(ev_c)),
        },
        "diffusion": {
            "dispersion_C12_over_C11": round(ratio_diff, 3),
            "ci95": [round(x, 3) for x in ci_diff],
            "interpretation": "~1 => equal local variability; >1 => C12 "
            "evolution more variable around its own field",
            "n_11": int(m11.sum()),
            "n_12": int(m12.sum()),
        },
        "trajectory_geometry_descriptive": geo,
        "population": files_note,
    }
    out = ROOT / "outputs" / "continuous_state_decomposition.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                "drift_ratio": report["drift"]["ratio_cross_over_within"],
                "ci": report["drift"]["ratio_ci95"],
                "diffusion_ratio": report["diffusion"]["dispersion_C12_over_C11"],
                "ci": report["diffusion"]["ci95"],
                "geometry": geo,
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
