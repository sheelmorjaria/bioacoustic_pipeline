#!/usr/bin/env python3
"""Predictive profile: horizon decay + order-specificity (prespecified).

(1) HORIZON DECAY — how far into the future does history remain useful?
Horizons prespecified at 40/80/160/320 ms (h = 4, 8, 16, 32 frames);
at each, Delta_history(h) = L(state-only) - L(history-aware) with the
established ridge machinery; block-clustered bootstrap.

(2) ORDER-SPECIFICITY — does true temporal order beat matched shuffled
histories? At h = 40 ms the history states are replaced by random
same-trajectory states (occupancy preserved, adjacency destroyed).
Delta_order = L(shuffled) - L(true): positive => order carries
predictive structure beyond occupancy; ~0 => continuity/occupancy.
Never a token/grammar claim.

Usage:
    python scripts/predictive_profile.py

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

HORIZONS = (4, 8, 16, 32)
LAGS = (4, 8, 12, 20)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
BOOT_B = 1000
SEED = 0
TRAIN_FRAC = 0.7
VAL_FRAC = 0.25  # of fit side, for alpha selection


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
        MaskProvenance,
        ScientificRunContract,
        assert_feature_contract,
        raise_for_science,
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
            if valid.sum() >= 40:
                bouts.append(
                    {"file": nm, "block": filename_block_group(nm), "z": f, "valid": valid}
                )
    blocks = sorted({b["block"] for b in bouts})
    fit_blocks = {
        bl
        for bl in blocks
        if int(hashlib.sha1(bl.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    for b in bouts:
        b["part"] = "fit" if b["block"] in fit_blocks else "eval"
    assert_feature_contract(bouts[0]["z"], context="profile fitting")
    print(f"bouts: {len(bouts)}", flush=True)

    def _collect(h, shuffle=False):
        X, Y, BLK = [], [], []
        rng = np.random.RandomState(SEED)
        for b in bouts:
            z, v = b["z"], b["valid"]
            T = len(z)
            pool = np.where(v)[0]
            if len(pool) < 25:
                continue
            for t in range(max(LAGS) + 1, T - h):
                if not (v[t] and v[t - 1] and v[t + h] and all(v[t - k] for k in LAGS)):
                    continue
                if shuffle:
                    # 1 substitute for s_{t-1} + one per older lag (5 total)
                    picks = [pool[rng.randint(0, len(pool))] for _ in range(1 + len(LAGS))]
                    hist = [z[t]] + [z[p] for p in picks]
                else:
                    hist = [z[t], z[t - 1]] + [z[t - k] for k in LAGS]
                X.append(np.concatenate(hist))
                Y.append(z[t + h])
                BLK.append(b["block"])
        return np.stack(X), np.stack(Y), np.array(BLK)

    def _loss(r):
        return float(np.sqrt((r**2).mean(0)).mean())

    files = sorted({b["file"] for b in bouts})
    part = {b["file"]: b["part"] for b in bouts}
    contract = ScientificRunContract(
        run_name="predictive_profile",
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
                produced_by="valid t-1,t,t+h and lags",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest={f: {"group": f, "part": part[f]} for f in files},
        eval_split={"provenance": "recording-block hash 70/30", "unit": "recording block"},
    )
    enforcement = raise_for_science(contract)

    # ---- three-way split: fit / val(alpha) / eval, by recording block
    rng_split = np.random.RandomState(SEED)
    fit_list = sorted(fit_blocks)
    rng_split.shuffle(fit_list)
    val_blocks = set(fit_list[: max(1, int(len(fit_list) * VAL_FRAC))])
    fit_only = set(fit_list) - val_blocks

    # ---- COMMON positions eligible at ALL horizons (so the decay curve
    # reflects horizon, not changing evaluation populations)
    def _eligible_all():
        out = []
        hmax = max(HORIZONS)
        for bi, b in enumerate(bouts):
            z, v = b["z"], b["valid"]
            T = len(z)
            for t in range(max(LAGS) + 1, T - hmax):
                if v[t] and v[t - 1] and v[t + hmax] and all(v[t - k] for k in LAGS):
                    out.append((bi, t))
        return out

    common_pos = _eligible_all()
    print(f"common positions (eligible at ALL horizons): {len(common_pos)}", flush=True)

    def _collect_common(h, shuffle=False):
        X, Y, BLK = [], [], []
        rng = np.random.RandomState(SEED)
        for bi, t in common_pos:
            b = bouts[bi]
            z, v = b["z"], b["valid"]
            pool = np.where(v)[0]
            if shuffle:
                picks = [pool[rng.randint(0, len(pool))] for _ in range(1 + len(LAGS))]
                hist = [z[t]] + [z[p] for p in picks]
            else:
                hist = [z[t], z[t - 1]] + [z[t - k] for k in LAGS]
            X.append(np.concatenate(hist))
            Y.append(z[t + h])
            BLK.append(b["block"])
        return np.stack(X), np.stack(Y), np.array(BLK)

    decay_common = {}
    for h in HORIZONS:
        X, Y, BLK = _collect_common(h)
        mu_x, sd_x = X.mean(0), X.std(0) + 1e-9
        mu_y, sd_y = Y.mean(0), Y.std(0) + 1e-9
        Xn, Yn = (X - mu_x) / sd_x, (Y - mu_y) / sd_y
        is_fit = np.isin(BLK, sorted(fit_only))
        is_val = np.isin(BLK, sorted(val_blocks))
        ev = ~is_fit & ~is_val
        bs = bf = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(Xn[is_fit][:, :12], Yn[is_fit], alpha)
            lv = _loss(ridge_predict(w, b, Xn[is_val][:, :12]) - Yn[is_val])
            if lv < bs[1]:
                bs = (alpha, lv)
            w, b = ridge_fit(Xn[is_fit], Yn[is_fit], alpha)
            lv = _loss(ridge_predict(w, b, Xn[is_val]) - Yn[is_val])
            if lv < bf[1]:
                bf = (alpha, lv)
        ws, bs_ = ridge_fit(Xn[is_fit][:, :12], Yn[is_fit], bs[0])
        wf, bf_ = ridge_fit(Xn[is_fit], Yn[is_fit], bf[0])
        ev_pos = np.where(ev)[0]
        rs = ridge_predict(ws, bs_, Xn[ev][:, :12]) - Yn[ev]
        rf = ridge_predict(wf, bf_, Xn[ev]) - Yn[ev]
        by = {}
        for j, i in enumerate(ev_pos):
            by.setdefault(BLK[i], []).append(j)
        keys = sorted(by)
        brng = np.random.RandomState(SEED)
        reps = []
        for _ in range(BOOT_B):
            pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
            idx = np.concatenate([by[g] for g in pick])
            reps.append(_loss(rs[idx]) - _loss(rf[idx]))
        decay_common[str(h * 10)] = {
            "delta_history": round(_loss(rs) - _loss(rf), 4),
            "ci95": [
                round(float(np.percentile(reps, 2.5)), 4),
                round(float(np.percentile(reps, 97.5)), 4),
            ],
            "n_eval": int(ev.sum()),
            "alpha_state": bs[0],
            "alpha_history": bf[0],
            "L_state": round(_loss(rs), 4),
            "L_history": round(_loss(rf), 4),
        }
        print(
            f"[common] h={h * 10}ms: delta "
            f"{decay_common[str(h * 10)]['delta_history']:+.4f} "
            f"CI {decay_common[str(h * 10)]['ci95']}",
            flush=True,
        )

    Xt, Yt, BLKt = _collect_common(4, shuffle=False)
    Xs, _, _ = _collect_common(4, shuffle=True)
    n_common = min(len(Xt), len(Xs))
    Xt, Yt, BLKt = Xt[:n_common], Yt[:n_common], BLKt[:n_common]
    Xs = Xs[:n_common]
    Xt, Yt, BLKt = _collect(4, shuffle=False)
    Xs, _, _ = _collect(4, shuffle=True)
    n_common = min(len(Xt), len(Xs))
    Xt, Yt, BLKt = Xt[:n_common], Yt[:n_common], BLKt[:n_common]
    Xs = Xs[:n_common]
    mu_x, sd_x = Xt.mean(0), Xt.std(0) + 1e-9
    mu_y, sd_y = Yt.mean(0), Yt.std(0) + 1e-9
    Xtn, Ytn = (Xt - mu_x) / sd_x, (Yt - mu_y) / sd_y
    Xsn = (Xs - mu_x) / sd_x
    is_fit = np.isin(BLKt, sorted(fit_only))
    is_val = np.isin(BLKt, sorted(val_blocks))
    ev = ~is_fit & ~is_val
    best = (None, np.inf)
    for alpha in ALPHAS:
        w, b = ridge_fit(Xtn[is_fit], Ytn[is_fit], alpha)
        lv = _loss(ridge_predict(w, b, Xtn[is_val]) - Ytn[is_val])
        if lv < best[1]:
            best = (alpha, lv)
    w, b = ridge_fit(Xtn[is_fit], Ytn[is_fit], best[0])
    ev_pos = np.where(ev)[0]
    r_true = ridge_predict(w, b, Xtn[ev]) - Ytn[ev]
    r_shuf = ridge_predict(w, b, Xsn[ev]) - Ytn[ev]
    by = {}
    for j, i in enumerate(ev_pos):
        by.setdefault(BLKt[i], []).append(j)
    keys = sorted(by)
    brng = np.random.RandomState(SEED)
    reps = []
    for _ in range(BOOT_B):
        pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
        idx = np.concatenate([by[g] for g in pick])
        reps.append(_loss(r_shuf[idx]) - _loss(r_true[idx]))
    order = {
        "delta_order": round(_loss(r_shuf) - _loss(r_true), 4),
        "ci95": [
            round(float(np.percentile(reps, 2.5)), 4),
            round(float(np.percentile(reps, 97.5)), 4),
        ],
        "L_true": round(_loss(r_true), 4),
        "L_shuffled": round(_loss(r_shuf), 4),
        "n_eval": int(ev.sum()),
        "interpretation": "positive => temporal order carries predictive "
        "structure beyond state occupancy; ~0 => continuity/occupancy. "
        "Never a token/grammar claim.",
    }

    report = {
        "study": "predictive profile: horizon decay + order-specificity "
        "(prespecified horizons 40/80/160/320 ms)",
        "release_enforcement": enforcement,
        "horizon_decay_common_positions": decay_common,
        "order_specificity": order,
        "framing": "temporal predictability measurements — no memory, no "
        "grammar, no meaning claims",
    }
    out = ROOT / "outputs" / "predictive_profile.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(order, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
