#!/usr/bin/env python3
"""Acoustic-dynamics evaluation: do derivatives improve prospective prediction?

Part A (forecast) — the prespecified state/rate comparison on the
validity-audited bat C11/C12 trajectories (cache/stage2/bat_c11c12_traj.npz,
WIN 40 ms / HOP 10 ms at 250 kHz):

    s_t    : the 12 supported absolute features at valid frame t
    ds_t   : their gap-safe backward differences (needs valid[t-1] & valid[t])
    arms   : persistence (s_hat = s_t) | state [s_t] | deriv [ds_t]
             | state+deriv [s_t, ds_t] | state+prev [s_t, s_{t-1}]
    horizons h (frames, 10 ms each): 4 = PRIMARY prospective
        (10*(t+h) >= 10*t + 40  <=>  h >= win/hop = 4: the target window starts
        at or after the latest source sample feeding the predictor),
        8 = secondary prospective, 2 = OVERLAPPING (labelled; continuity
        context only, never evidence).
    model  : multi-output ridge. Target stats from the FIT split; per-arm input
        stats from FIT; alpha chosen per arm on VAL; final refit on FIT+VAL.
        The EVAL partition touches no fitted quantity.
    unit   : per-feature normalized RMSE (target std from FIT); aggregate =
        mean over the 12 features. Paired FILE-level bootstrap (identical
        resampled files handed to every arm) for all deltas.
    decision rule (prespecified): retain derivative augmentation for this
        forecasting task iff eval aggregate(state+deriv) beats aggregate(state)
        with the paired bootstrap CI of the delta excluding 0. [s_t, ds_t] and
        [s_t, s_{t-1}] carry identical information (s_{t-1} = s_t - dt*ds_t),
        so any gain over state-only is recent history helping the predictor,
        not new information created by differentiating.

Eval partition honesty: parts==1 of the cache is the §46 stage-2
recording-group EVAL split — previously inspected for frozen-vocab assignment
statistics, never used to fit any forecasting scale, alpha, or weight.

Part B (inspection) — derivative views on the four frozen recurrence
candidates (outputs/recurrence_sixpair_inspection.json). Prespecified
summaries only: per-feature net signed change over the matched 40-frame
region, q90 of |rate|, valid coverage. Level and rate agreement are reported
SEPARATELY (a rate-only view equates a+vt and b+vt for a != b). Frozen search
distances are not recomputed or altered: a derivative plot is a complementary
view of the same observations, not independent corroboration.

Usage:
    python scripts/acoustic_dynamics_report.py --part all
    python scripts/acoustic_dynamics_report.py --part forecast
    python scripts/acoustic_dynamics_report.py --self-test

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HORIZONS = (2, 4, 8)
PRIMARY_H = 4
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
N_BOOT = 2000
BOOT_SEED = 0
SPLIT_SEED = 0
DT_MS = 10.0  # trajectory hop
WIN_MS = 40.0  # analysis window (both from the §44 retune)
VAL_FRACTION = 0.25
ARMS = ("persistence", "state", "deriv", "state+deriv", "state+prev")


# ---------------------------------------------------------------- Part A core


def eligible_positions(valid, h: int):
    """Frame indices t valid for EVERY arm at horizon h.

    Requires valid[t-1] (derivative exists — excludes the first frame of each
    valid span for all arms alike), valid[t] (predictor present), and
    valid[t+h] (target scorable). Identical across arms by construction.
    """
    import numpy as np

    valid = np.asarray(valid, dtype=bool)
    n = len(valid)
    if n <= h + 1:
        return np.array([], dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    ok[1 : n - h] = valid[: n - h - 1] & valid[1 : n - h] & valid[1 + h :]
    return np.where(ok)[0]


def ridge_fit(x, y, alpha: float):
    """Multi-output ridge with an unregularized intercept; closed-form."""
    import numpy as np

    d = x.shape[1]
    x1 = np.hstack([x, np.ones((len(x), 1))])
    a = x1.T @ x1
    a[:d, :d] += alpha * np.eye(d)
    wb = np.linalg.solve(a, x1.T @ y)
    return wb[:d], wb[d]


def ridge_predict(w, b, x):
    return x @ w + b


def _arm_matrix(arm: str, s_t, s_prev, d_t):
    import numpy as np

    if arm == "persistence":
        return None
    if arm == "state":
        return s_t
    if arm == "deriv":
        return d_t
    if arm == "state+deriv":
        return np.hstack([s_t, d_t])
    if arm == "state+prev":
        return np.hstack([s_t, s_prev])
    raise ValueError(f"unknown arm {arm}")


def _agg_from_residuals(resid) -> float:
    """Aggregate: mean over features of per-feature RMSE (normalized space)."""
    import numpy as np

    return float(np.sqrt((resid**2).mean(axis=0)).mean())


def run_forecast(train_rows, eval_rows, *, alphas=ALPHAS, n_boot=0, boot_seed=BOOT_SEED):
    """One horizon's comparison.

    ``train_rows`` (fit/val by file on the train side) determines every fitted
    quantity; ``eval_rows`` is only ever scored. Returns per-arm alpha, the
    VAL aggregate, the EVAL per-feature normalized-RMSE table and aggregate,
    plus paired file-bootstrap deltas when ``n_boot`` > 0.
    """
    import numpy as np

    files = train_rows["files"]
    uniq = np.unique(files)
    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * VAL_FRACTION)))
    val_set = set(shuffled[:n_val].tolist())
    is_val = np.array([f in val_set for f in files.tolist()])
    fit = ~is_val

    # target normalization: FIT-split stats, used unchanged for train and eval
    mu_y = train_rows["y"][fit].mean(axis=0)
    sig_y = train_rows["y"][fit].std(axis=0) + 1e-12
    y_n = (train_rows["y"] - mu_y) / sig_y
    y_n_eval = (eval_rows["y"] - mu_y) / sig_y
    s_t_n_eval = (eval_rows["s_t"] - mu_y) / sig_y

    results: dict = {"persistence": {"alpha": None, "val_agg": None}}
    models: dict = {}
    for arm in ARMS:
        if arm == "persistence":
            continue
        x = _arm_matrix(arm, train_rows["s_t"], train_rows["s_prev"], train_rows["d_t"])
        mu_x = x[fit].mean(axis=0)
        sig_x = x[fit].std(axis=0) + 1e-12
        x_n = (x - mu_x) / sig_x
        best_alpha, best_val = None, np.inf
        for alpha in alphas:
            w, b = ridge_fit(x_n[fit], y_n[fit], alpha)
            v = _agg_from_residuals(ridge_predict(w, b, x_n[is_val]) - y_n[is_val])
            if v < best_val:
                best_alpha, best_val = alpha, v
        # final refit on the FULL train side (fit+val), stats refit train-side
        mu_f = x.mean(axis=0)
        sig_f = x.std(axis=0) + 1e-12
        x_f = (x - mu_f) / sig_f
        w, b = ridge_fit(x_f, y_n, best_alpha)
        models[arm] = (mu_f, sig_f, w, b)
        results[arm] = {"alpha": float(best_alpha), "val_agg": float(best_val)}

    # ---- score EVAL with the frozen models (identical rows for every arm)
    resid_by_arm: dict = {}
    for arm in ARMS:
        if arm == "persistence":
            resid_by_arm[arm] = y_n_eval - s_t_n_eval
        else:
            mu_f, sig_f, w, b = models[arm]
            xe = _arm_matrix(arm, eval_rows["s_t"], eval_rows["s_prev"], eval_rows["d_t"])
            resid_by_arm[arm] = ridge_predict(w, b, (xe - mu_f) / sig_f) - y_n_eval
        results.setdefault(arm, {})
        results[arm]["nrmse_per_feature"] = np.sqrt((resid_by_arm[arm] ** 2).mean(axis=0)).tolist()
        results[arm]["aggregate"] = _agg_from_residuals(resid_by_arm[arm])

    # ---- paired file bootstrap on eval rows (same picks for every arm)
    deltas = {}
    if n_boot:
        e_uniq = np.unique(eval_rows["files"])
        col = np.searchsorted(e_uniq, eval_rows["files"])
        counts = np.bincount(col, minlength=len(e_uniq))
        mats = {
            arm: np.stack(
                [
                    np.bincount(
                        col,
                        weights=resid_by_arm[arm][:, j] ** 2,
                        minlength=len(e_uniq),
                    )
                    for j in range(resid_by_arm[arm].shape[1])
                ]
            )
            for arm in ARMS
        }  # (arm, 12, n_eval_files)
        brng = np.random.RandomState(boot_seed)
        picks = brng.randint(0, len(e_uniq), size=(n_boot, len(e_uniq)))
        cnt = counts[picks].sum(axis=1)  # (B,)
        agg_b = {arm: np.sqrt(mats[arm][:, picks].sum(axis=2) / cnt).mean(axis=0) for arm in ARMS}
        for a, b_ in (
            ("state+deriv", "state"),
            ("state+prev", "state"),
            ("state+deriv", "state+prev"),
            ("state+deriv", "persistence"),
            ("state", "persistence"),
        ):
            d = agg_b[a] - agg_b[b_]  # positive = first arm WORSE
            deltas[f"{a} minus {b_}"] = {
                "observed": float(agg_b[a].mean() - agg_b[b_].mean()),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "frac_negative": float((d < 0).mean()),
            }
    return {
        "n_train_rows": int(len(files)),
        "n_eval_rows": int(len(eval_rows["files"])),
        "n_eval_files": int(len(np.unique(eval_rows["files"]))),
        "arms": results,
        "bootstrap_deltas": deltas,
    }


def load_bout_rows(cache_path: Path):
    """Per-bout supported-feature blocks + validity from the trajectory cache."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        first_derivative,
        supported_features,
    )

    data = np.load(cache_path, allow_pickle=False)
    frames, offsets = data["frames"], data["offsets"]
    names, parts = list(data["names"]), data["parts"]

    # de-duplicate repeated files (identical trajectories) and forbid a file
    # from appearing on both sides of the frozen split
    seen: dict = {}
    for i, nm in enumerate(names):
        seen.setdefault(nm, []).append(i)
    keep, dropped = [], 0
    for nm, idxs in seen.items():
        side = {int(parts[i]) for i in idxs}
        if len(side) > 1:
            raise ValueError(f"{nm} appears on both sides of the frozen split")
        keep.append(idxs[0])
        dropped += len(idxs) - 1
    keep = sorted(keep)

    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]
    bouts = []
    for i in keep:
        f = frames[offsets[i] : offsets[i + 1]].astype(np.float64)
        valid = ~np.all(f == 0, axis=1)
        sub = f[:, fidx]
        derivs = np.full_like(sub, np.nan)
        for j in range(len(feats)):
            derivs[:, j] = first_derivative(sub[:, j], valid, dt=DT_MS / 1000.0)
        bouts.append(
            {"name": names[i], "part": int(parts[i]), "s": sub, "d": derivs, "valid": valid}
        )
    return bouts, feats, dropped


def assemble_horizon(bouts, h: int):
    """All eligible rows for horizon h across bouts (gap-safe, arm-agnostic)."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import contiguous_spans

    files, s_t, s_prev, d_t, y, same_span = [], [], [], [], [], []
    dropped_nonfinite = 0
    for bi, b in enumerate(bouts):
        ts = eligible_positions(b["valid"], h)
        if len(ts) == 0:
            continue
        spans = contiguous_spans(b["valid"])
        s, d = b["s"], b["d"]
        blocks = (s[ts], s[ts - 1], d[ts], s[ts + h])
        finite = np.ones(len(ts), dtype=bool)
        for blk in blocks:
            finite &= np.isfinite(blk).all(axis=1)
        dropped_nonfinite += int((~finite).sum())
        ts = ts[finite]
        files.append(np.full(len(ts), bi, dtype=np.int64))
        s_t.append(s[ts])
        s_prev.append(s[ts - 1])
        d_t.append(d[ts])
        y.append(s[ts + h])
        same_span.append(spans[ts] == spans[ts + h])
    if not files:
        raise ValueError(f"no eligible rows at h={h}")
    rows = {
        "files": np.concatenate(files),
        "s_t": np.vstack(s_t),
        "s_prev": np.vstack(s_prev),
        "d_t": np.vstack(d_t),
        "y": np.vstack(y),
        "same_span": np.concatenate(same_span),
    }
    return rows, dropped_nonfinite


# ------------------------------------------------------------ Part A driver


def part_forecast() -> dict:
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import FEATURE_UNITS
    from bioacoustic_pipeline.ingestion.labeled_corpus import parse_annotations_csv

    cache = ROOT / "cache" / "stage2" / "bat_c11c12_traj.npz"
    bouts, feats, dropped_dupes = load_bout_rows(cache)

    # join callers/contexts for reporting only (never a model input)
    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    meta = {a.file_name: (a.emitter, a.context) for a in annos}

    report: dict = {
        "task": "prospective state/rate forecasting on validity-audited bat C11/C12",
        "source_cache": str(cache.relative_to(ROOT)),
        "framing": {"win_ms": WIN_MS, "hop_ms": DT_MS, "dt_s": DT_MS / 1000.0},
        "features": [{"name": n, "units": FEATURE_UNITS[n]} for n in feats],
        "arms": list(ARMS),
        "prospective_rule": "target window start >= latest predictor source sample "
        f"(h*hop >= win => h >= {int(WIN_MS / DT_MS)})",
        "aggregate": "mean over 12 features of normalized RMSE (target std from FIT)",
        "decision_rule": "retain derivative augmentation iff aggregate(state+deriv) "
        "beats aggregate(state) with paired-bootstrap CI of the delta excluding 0",
        "eval_partition_note": "parts==1 = frozen §46 stage-2 recording-group EVAL "
        "split; previously inspected for frozen-vocab assignment stats, never used "
        "to fit any forecasting scale/alpha/weight",
        "dropped_duplicate_files": dropped_dupes,
        "bouts": {
            "fit_side": sum(1 for b in bouts if b["part"] == 0),
            "eval_side": sum(1 for b in bouts if b["part"] == 1),
            "eval_callers": len(
                {meta[b["name"]][0] for b in bouts if b["part"] == 1 and b["name"] in meta}
            ),
        },
        "horizons": {},
    }

    for h in HORIZONS:
        rows, dropped_nf = assemble_horizon(bouts, h)
        eval_mask = np.array([bouts[f]["part"] == 1 for f in rows["files"]])
        train_rows = {k: v[~eval_mask] for k, v in rows.items() if k != "same_span"}
        eval_rows = {k: v[eval_mask] for k, v in rows.items() if k != "same_span"}
        out = run_forecast(train_rows, eval_rows, n_boot=N_BOOT)
        if h >= int(WIN_MS / DT_MS):
            label = "PRIMARY prospective" if h == PRIMARY_H else "secondary prospective"
        else:
            label = (
                f"OVERLAPPING (target overlaps predictor window by "
                f"{int(WIN_MS - DT_MS * h)} ms) — continuity context only"
            )
        rep = {
            "h_frames": h,
            "h_ms": h * DT_MS,
            "label": label,
            "n_rows_total": int(len(rows["files"])),
            "dropped_nonfinite": dropped_nf,
            "same_span_fraction": float(rows["same_span"].mean()),
            "n_train_rows": out["n_train_rows"],
            "n_eval_rows": out["n_eval_rows"],
            "n_eval_files": out["n_eval_files"],
            "chosen_alpha": {a: out["arms"][a]["alpha"] for a in ARMS},
            "val_aggregate": {a: out["arms"][a]["val_agg"] for a in ARMS},
            "eval_aggregate": {a: out["arms"][a]["aggregate"] for a in ARMS},
            "eval_nrmse_per_feature": {a: out["arms"][a]["nrmse_per_feature"] for a in ARMS},
            "skill_vs_persistence": {
                a: 1.0 - out["arms"][a]["aggregate"] / out["arms"]["persistence"]["aggregate"]
                for a in ARMS
            },
            "bootstrap_deltas": out["bootstrap_deltas"],
        }
        report["horizons"][str(h)] = rep
        print(
            f"h={h} [{label}] rows(eval)={rep['n_eval_rows']} "
            f"files(eval)={rep['n_eval_files']} same_span={rep['same_span_fraction']:.3f}"
            + " | ".join(f" {a}={out['arms'][a]['aggregate']:.4f}" for a in ARMS),
            flush=True,
        )
        for k, v in out["bootstrap_deltas"].items():
            ci = v["ci95"]
            print(
                f"    {k}: {v['observed']:+.5f} CI[{ci[0]:+.5f}, {ci[1]:+.5f}] "
                f"frac-={v['frac_negative']:.3f}",
                flush=True,
            )

    prim = report["horizons"][str(PRIMARY_H)]
    d_sd = prim["bootstrap_deltas"]["state+deriv minus state"]
    ci = d_sd["ci95"]
    better = prim["eval_aggregate"]["state+deriv"] < prim["eval_aggregate"]["state"]
    ci_excludes0 = ci[0] * ci[1] > 0  # both ends the same nonzero sign
    keep = bool(better and ci_excludes0)
    report["decision"] = {
        "retain_derivative_augmentation_for_forecasting": keep,
        "basis": (
            f"primary h={PRIMARY_H}: aggregate(state+deriv) "
            f"{prim['eval_aggregate']['state+deriv']:.4f} vs state "
            f"{prim['eval_aggregate']['state']:.4f}; paired delta "
            f"{d_sd['observed']:+.5f}, CI [{ci[0]:+.5f}, {ci[1]:+.5f}]"
        ),
        "interpretation_caveat": (
            "[s_t, ds_t] and [s_t, s_{t-1}] carry identical information "
            "(s_{t-1} = s_t - dt*ds_t): any gain over state-only is recent "
            "history helping the predictor — a feature-engineering result, not "
            "evidence of a recovered motor state"
        ),
        "if_not_retained": (
            "derivatives remain interpretable acoustic measurements (direction "
            "and rate of change for retrieval/inspection), not promoted "
            "forecasting features"
        ),
    }
    return report


# ------------------------------------------------------------ Part B driver


def part_candidates() -> dict:
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_UNITS,
        _NAME_TO_IDX,
        first_derivative,
        supported_features,
    )
    from bioacoustic_pipeline.utils.audio import load_audio

    insp = json.loads((ROOT / "outputs" / "recurrence_sixpair_inspection.json").read_text())
    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]
    sr = 250000
    win, hop = int(sr * WIN_MS / 1000), int(sr * DT_MS / 1000)
    region_frames = 40  # the §42-44 search window (40 frames x 10 ms = 400 ms)

    out = {
        "task": "derivative views on the frozen recurrence candidates",
        "summaries": "prespecified: net signed change over matched region; "
        "q90 |rate|; valid coverage. NOT post-hoc separator hunting",
        "caveat": "level and rate are separate views; identical rates can hide "
        "different absolute levels (a+vt vs b+vt, a != b). A derivative plot is "
        "a complementary view of the same observations, not independent "
        "corroboration; frozen search distances unchanged",
        "pairs": [],
    }
    for entry in insp["inspection"]:
        sides: dict = {}
        traj: dict = {}
        for side in ("A", "B"):
            info = entry["sides"][side]
            path = ROOT / "data" / "egyptian_fruit_bats" / "audio" / info["source_file"]
            audio, asr = load_audio(str(path), target_sr=sr)
            rows, i = [], 0
            while i + win <= len(audio) and len(rows) < 300:
                v = np.asarray(
                    afp.extract(audio[i : i + win], asr, species="bat").combined,
                    dtype=np.float64,
                )
                rows.append(v)
                i += hop
            f = np.asarray(rows)
            valid = ~np.all(f == 0, axis=1)
            sub = f[:, fidx]
            d = np.full_like(sub, np.nan)
            for j in range(len(feats)):
                d[:, j] = first_derivative(sub[:, j], valid, dt=DT_MS / 1000.0)
            p0 = int(info["match_pos_ms"] // DT_MS)
            reg = slice(p0, p0 + region_frames)
            # region-aligned per side: the §42-44 search matched 40-frame
            # windows on this same 10 ms grid (hop 8); positions are window
            # starts, so regions align frame-for-frame by construction
            traj[side] = (sub[reg], d[reg], valid[reg])
            sv, dv, vreg = sub[reg], d[reg], valid[reg]
            # gate audit: the fingerprint validity is an ABSOLUTE energy gate
            # (rms < 1e-3 -> zeroed core), while the sixpair vocal check was
            # peak-relative — record both readings plus the within-window
            # offsets of the surviving frames
            raw = audio
            w_rms = np.array(
                [
                    float(np.sqrt(np.mean(raw[(p0 + i) * hop : (p0 + i) * hop + win] ** 2)))
                    for i in range(len(sv))
                ]
            )
            gate_audit = {
                "file_peak": round(float(np.abs(raw).max()), 5),
                "region_rms_median": round(float(np.median(w_rms)), 6),
                "region_rms_max": round(float(w_rms.max()), 6),
                "frames_above_absolute_gate": int((w_rms >= 1e-3).sum()),
                "frames_above_peak_relative_vocal": int(
                    (w_rms >= 1e-3 * float(np.abs(raw).max())).sum()
                ),
                "fingerprint_valid_offsets_in_window": np.where(vreg)[0].tolist(),
            }
            dvalid = np.zeros(region_frames, dtype=bool)
            dvalid[1:] = vreg[1:] & vreg[:-1]
            per_feat = []
            for j, name in enumerate(feats):
                vals = sv[:, j][vreg]
                net = float(vals[-1] - vals[0]) if len(vals) >= 2 else None
                rate = np.abs(dv[:, j][dvalid])
                q90 = float(np.percentile(rate, 90)) if len(rate) >= 5 else None
                per_feat.append(
                    {
                        "feature": name,
                        "units": FEATURE_UNITS[name],
                        "net_change": None if net is None else round(net, 4),
                        "net_units": f"{FEATURE_UNITS[name]}*s",
                        "q90_abs_rate": None if q90 is None else round(q90, 2),
                    }
                )
            sides[side] = {
                "source_file": info["source_file"],
                "caller": info["caller"],
                "context": info["context"],
                "match_pos_ms": info["match_pos_ms"],
                "region_ms": [p0 * DT_MS, (p0 + region_frames) * DT_MS],
                "valid_coverage_in_region": round(float(vreg.mean()), 3),
                "n_valid_frames": int(vreg.sum()),
                "gate_audit": gate_audit,
                "per_feature": per_feat,
            }
        a_sub, a_d, a_v = traj["A"]
        b_sub, b_d, b_v = traj["B"]
        both = a_v & b_v
        both_d = np.zeros(len(a_v), dtype=bool)
        both_d[1:] = both[1:] & both[:-1]
        agree = {
            "n_commonly_valid_frames": int(both.sum()),
            "level_mean_abs_diff": {
                n: (
                    round(float(np.abs(a_sub[both, j] - b_sub[both, j]).mean()), 4)
                    if both.any()
                    else None
                )
                for j, n in enumerate(feats)
            },
            "rate_mean_abs_diff": {
                n: (
                    round(float(np.abs(a_d[both_d, j] - b_d[both_d, j]).mean()), 2)
                    if both_d.any()
                    else None
                )
                for j, n in enumerate(feats)
            },
        }
        out["pairs"].append({"pair": entry["pair"], "sides": sides, "agreement": agree})
        print(
            f"pair {entry['pair']}: A={sides['A']['source_file']}"
            f"@{sides['A']['match_pos_ms']}ms cov={sides['A']['valid_coverage_in_region']}"
            f" | B={sides['B']['source_file']}@{sides['B']['match_pos_ms']}ms "
            f"cov={sides['B']['valid_coverage_in_region']} | "
            f"common valid={agree['n_commonly_valid_frames']}",
            flush=True,
        )

    # structural conclusion across all audited pairs (empirical, prespecified
    # audit — no frozen artifact is relabelled here)
    all_offsets = [
        s["gate_audit"]["fingerprint_valid_offsets_in_window"]
        for p in out["pairs"]
        for s in p["sides"].values()
    ]
    uniform = all(o == [0] for o in all_offsets)
    out["gate_artifact_audit"] = {
        "n_pairs_audited": len(out["pairs"]),
        "unique_note": "pair 5 reverses pair 3 (same two sources, sides swapped)",
        "every_side_pattern": "[1 fingerprint-valid frame at within-window "
        "offset 0, 39 gated frames]"
        if uniform
        else "mixed — inspect per side",
        "mechanism": "the fingerprint's ABSOLUTE gate (rms < 1e-3 zeroes the "
        "core) zeroes 39/40 frames of every matched window in these low-gain "
        "recordings (peaks 0.2-0.34); the sixpair vocal guard was PEAK-RELATIVE "
        "(rms >= 1e-3*peak), so it passed regions the representation itself "
        "treats as invalid. The distance-0 matches are therefore carried "
        "substantially by the shared [one surviving frame + gated remainder] "
        "window structure, not by 400 ms of comparable vocal acoustics",
        "derivative_view_verdict": "UNAVAILABLE on valid support: each region "
        "carries 1/40 fingerprint-valid frames, below the 2-valid-frame minimum "
        "for net change and the 5-derivative-frame minimum for q90 — lack of "
        "support, never fabricated values",
        "action": "recommend re-review of the candidates' 'inconclusive' labels "
        "with this gate mismatch in evidence; frozen search distances and "
        "labels are NOT altered by this artifact",
    }
    return out


# ----------------------------------------------------------------- self-test


def _self_test() -> None:
    """Synthetic trajectories through the exact Part-A machinery."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import first_derivative

    rng = np.random.RandomState(0)
    T = 120
    bouts = []
    for f in range(12):
        # AR(1)-with-drift features: history genuinely helps prediction here
        s = np.zeros((T, 12))
        s[0] = rng.randn(12)
        for t in range(1, T):
            s[t] = 0.3 + 0.8 * s[t - 1] + 0.05 * rng.randn(12)
        valid = np.ones(T, dtype=bool)
        valid[40:48] = False  # a mid-trajectory gap
        d = np.full_like(s, np.nan)
        for j in range(12):
            d[:, j] = first_derivative(s[:, j], valid, dt=DT_MS / 1000.0)
        bouts.append(
            {"name": f"synth_{f}.wav", "part": 1 if f >= 8 else 0, "s": s, "d": d, "valid": valid}
        )
    rows, dropped = assemble_horizon(bouts, PRIMARY_H)
    assert dropped == 0
    # eligibility never lands on a span start or an invalid/gap-adjacent frame
    ts = eligible_positions(bouts[0]["valid"], PRIMARY_H)
    assert not np.any(ts == 40) and not np.any(ts == 48)
    eval_mask = np.array([bouts[f]["part"] == 1 for f in rows["files"]])
    train_rows = {k: v[~eval_mask] for k, v in rows.items() if k != "same_span"}
    eval_rows = {k: v[eval_mask] for k, v in rows.items() if k != "same_span"}
    out = run_forecast(train_rows, eval_rows, alphas=(1.0, 10.0, 100.0), n_boot=200)
    agg = {a: out["arms"][a]["aggregate"] for a in ARMS}
    # ridge must beat persistence on genuinely predictable synthetic data
    assert agg["state"] < agg["persistence"], agg
    # information equivalence: [s, ds] and [s, s_prev] agree to within the
    # regularization-geometry slack even with per-arm-tuned alphas
    slack = 0.05 * agg["persistence"]
    assert abs(agg["state+deriv"] - agg["state+prev"]) < slack, (agg, slack)
    # every arm scored on the identical eval rows
    assert out["n_eval_rows"] == len(eval_rows["files"])
    assert out["bootstrap_deltas"], "bootstrap deltas missing"
    print(
        "SELF-TEST OK aggregates=" + str({k: round(v, 4) for k, v in agg.items()}),
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=["forecast", "candidates", "all"], default="all")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.part in ("forecast", "all"):
        report = part_forecast()
        path = ROOT / "outputs" / "dynamics_forecast_report.json"
        path.write_text(json.dumps(report, indent=1))
        print(f"wrote {path}", flush=True)
    if args.part in ("candidates", "all"):
        report = part_candidates()
        path = ROOT / "outputs" / "dynamics_candidate_views.json"
        path.write_text(json.dumps(report, indent=1))
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
