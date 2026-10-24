#!/usr/bin/env python3
"""Dolphin whistle forecasting — WITHIN-RECORDING scope (explicitly limited).

SCOPE CAVEAT (review-standing, first in the artifact): the Di Nardo
release is 303 whistle clips extracted from ONE 98-minute recording.
Partitioning clips does NOT create independent recording sessions or
independent callers. This study is therefore a WITHIN-RECORDING
forecasting analysis under the frozen five-arm contract:

  - partition: TIME BLOCKS within the single recording. Clip numeric IDs
    are chronological proxies for extraction order; the fit/val/eval
    split is by ID tercile-blocks (declared proxy — the release does not
    ship source timestamps in the local annotations). The resampling
    unit is the ID block, NOT the clip: overlapping or adjacent whistles
    share recording context.
  - NO independent-session claim, NO caller/repertoire claim, NO
    species comparison. Result direction is informative about the
    frozen procedure on this material only.

Sample-rate decision (declared before scoring): clips are 192 kHz.
The frozen contract's physical grid is 40 ms window / 10 ms hop —
applied at 192 kHz natively (win 7680 / hop 1920), preserving the full
recorded band for the fingerprint's absolute gate; no resampling.
Median clip 0.47 s => most clips cannot host the 280 ms support extent;
INSUFFICIENT SUPPORT IS AN ACCEPTABLE TERMINAL OUTCOME (audit first).

Usage:
    python scripts/dolphin_forecasting.py

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

SR = 192000
REQUIRED_FRAMES = 28
BOOT_B = 2000
SEED = 0
MIN_TOTAL_POSITIONS = 1000  # smaller corpus: declared lower bar
N_BLOCKS = 12  # ID-tercile blocks for partitioning


def _extract(job):
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.utils.audio import load_audio

    path, name = job
    audio, sr = load_audio(str(path), target_sr=SR)
    win, hop = int(sr * 0.04), int(sr * 0.01)
    rows, i = [], 0
    while i * hop + win <= len(audio) and len(rows) < 700:
        rows.append(
            np.asarray(
                afp.extract(audio[i * hop : i * hop + win], sr, species="dolphin").combined,
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
        contiguous_spans,
        supported_features,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        assert_feature_contract,
        raise_for_science,
        source_manifest_from,
    )
    from history_context_forecasting import ALPHAS, H_MS, LAGS, _arm_matrix, _design

    raw = ROOT / "data" / "alp" / "dinardo2023_dolphin_whistles" / "v0.1.0" / "raw"
    wavs = sorted((raw / "audio").glob("*.wav"), key=lambda p: int(p.stem.split("_")[0]))
    print(f"clips: {len(wavs)} (one recording; within-recording scope)", flush=True)
    jobs = [(p, p.name) for p in wavs]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = [r for r in ex.map(_extract, jobs, chunksize=4) if r]
    fidx = [_NAME_TO_IDX[n] for n in supported_features()]

    clips = []
    n_short = 0
    for name, f44 in results:
        f = f44[:, fidx]
        valid = ~np.all(f == 0, axis=1)
        pos = []
        for sid in sorted(set(contiguous_spans(valid).tolist())):
            if sid < 0:
                continue
            idx = np.where(contiguous_spans(valid) == sid)[0]
            if len(idx) >= REQUIRED_FRAMES:
                lo, hi = int(idx[0]), int(idx[-1])
                pos.extend(range(lo + max(LAGS), hi - H_MS + 1))
        cid = int(name.split("_")[0])
        block = f"blk{cid * N_BLOCKS // 304:02d}"  # chronological proxy
        if pos:
            clips.append(
                {"name": name, "block": block, "id": cid, "frames": f, "valid": valid, "pos": pos}
            )
        else:
            n_short += 1
    positions_total = sum(len(c["pos"]) for c in clips)
    print(
        json.dumps(
            {
                "extracted": len(results),
                "no_support": n_short,
                "with_positions": len(clips),
                "positions": positions_total,
            },
            indent=1,
        ),
        flush=True,
    )

    audit = {
        "clips": len(wavs),
        "extracted": len(results),
        "no_support": n_short,
        "with_positions": len(clips),
        "positions": positions_total,
        "blocks": sorted({c["block"] for c in clips}),
        "scope": "WITHIN-RECORDING: one 98-min recording; ID-block "
        "partition is a chronological PROXY; no independent-"
        "session or caller claim",
        "bar": MIN_TOTAL_POSITIONS,
    }
    if positions_total < MIN_TOTAL_POSITIONS:
        out = ROOT / "outputs" / "dolphin_forecasting.json"
        out.write_text(
            json.dumps(
                {
                    "study": "dolphin whistle forecasting — WITHIN-RECORDING scope",
                    "audit": audit,
                    "outcome": "INSUFFICIENT SUPPORT — terminal; no models fitted",
                },
                indent=1,
            )
        )
        print("INSUFFICIENT SUPPORT — terminal", flush=True)
        return

    assert_feature_contract(clips[0]["frames"], context="dolphin fitting")
    rows = []
    for c in clips:
        track = {"z": c["frames"], "context": 11}
        for t in c["pos"]:
            x, u, y, _ = _design(track, t)
            rows.append({"x": x, "u": u, "y": y, "group": c["block"], "clip": c["name"]})
    blocks = sorted({r["group"] for r in rows})
    fit_blocks = {b for b in blocks if int(hashlib.sha1(b.encode()).hexdigest()[:8], 16) % 100 < 70}
    rng = np.random.RandomState(SEED)
    fit_list = sorted(fit_blocks)
    rng.shuffle(fit_list)
    val_blocks = set(fit_list[: max(1, int(len(fit_list) * 0.25))])
    fit_blocks = set(fit_list) - val_blocks
    tr = [r for r in rows if r["group"] in fit_blocks]
    va = [r for r in rows if r["group"] in val_blocks]
    ev = [r for r in rows if r["group"] not in fit_blocks | val_blocks]

    files = sorted({c["name"] for c in clips})
    part = {}
    for r in rows:
        part.setdefault(
            r["clip"],
            "fit" if r["group"] in fit_blocks else ("val" if r["group"] in val_blocks else "eval"),
        )
    contract = ScientificRunContract(
        run_name="dolphin_forecasting_within_recording",
        feature_schema_version=2,
        masks={
            "source_validity": MaskProvenance(
                kind="source_validity",
                produced_by="fingerprint absolute gate",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native 192 kHz",
            ),
            "feature_support": MaskProvenance(
                kind="feature_support",
                produced_by=f"contiguous spans >= {REQUIRED_FRAMES} frames",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native 192 kHz",
            ),
        },
        source_manifest=source_manifest_from(
            files,
            [part[f] for f in files],
            [f"blk{int(f.split('_')[0]) * N_BLOCKS // 304:02d}" for f in files],
        ),
        eval_split={
            "provenance": "WITHIN-RECORDING ID-block partition "
            "(chronological proxy; NOT independent sessions)",
            "unit": "ID block",
        },
    )
    enforcement = raise_for_science(contract)

    Ytr = np.stack([r["y"] for r in tr])
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-9
    Xtr_r = np.stack([r["x"] for r in tr])
    Utr_r = np.stack([r["u"] for r in tr])
    mu_x, sd_x = Xtr_r.mean(0), Xtr_r.std(0) + 1e-9
    mu_u, sd_u = Utr_r.mean(0), Utr_r.std(0) + 1e-9
    ev_names = {r["clip"] for r in ev}
    for c in clips:
        if c["name"] in ev_names:
            assert_feature_contract(c["frames"], context="dolphin scoring")
            break

    def _n(rs):
        X = (np.stack([r["x"] for r in rs]) - mu_x) / sd_x
        U = (np.stack([r["u"] for r in rs]) - mu_u) / sd_u
        Y = (np.stack([r["y"] for r in rs]) - mu) / sd
        return X, U, Y

    Xtr, Utr, Yn = _n(tr)
    Xva, Uva, Yva = _n(va)
    Xev, Uev, Yev = _n(ev)
    ones = {"tr": np.ones(len(tr)), "va": np.ones(len(va)), "ev": np.ones(len(ev))}

    arms, resid = {}, {}
    for arm in ("short", "shared"):
        A = _arm_matrix(arm, Xtr, Utr, ones["tr"])
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(A, Yn, alpha)
            loss_v = float(
                np.sqrt(
                    ((ridge_predict(w, b, _arm_matrix(arm, Xva, Uva, ones["va"])) - Yva) ** 2).mean(
                        axis=0
                    )
                ).mean()
            )
            if loss_v < best[1]:
                best = (alpha, loss_v)
        w, b = ridge_fit(A, Yn, best[0])
        pr = ridge_predict(w, b, _arm_matrix(arm, Xev, Uev, ones["ev"]))
        r_ = pr - Yev
        resid[arm] = r_
        arms[arm] = {"alpha": best[0], "loss": float(np.sqrt((r_**2).mean(0)).mean())}
    s_t = np.stack([r["x"][:12] for r in ev])
    pers = (s_t - mu) / sd - Yev
    arms["persistence"] = {"loss": float(np.sqrt((pers**2).mean(0)).mean())}
    arms["training_mean"] = {"loss": float(np.sqrt((Yev**2).mean(0)).mean())}
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
    by_g: dict = {}
    for i, r in enumerate(ev):
        by_g.setdefault(r["group"], []).append(i)
    keys = sorted(by_g)
    brng = np.random.RandomState(SEED)
    ci = {}
    for name, (ra, rb) in {
        "delta_older": ("short", "shared"),
        "delta_recent": ("state", "short"),
    }.items():
        reps = []
        for _ in range(BOOT_B):
            pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
            idx = np.concatenate([by_g[g] for g in pick])
            reps.append(_loss(resid[ra][idx]) - _loss(resid[rb][idx]))
        ci[name] = [
            round(float(np.percentile(reps, 2.5)), 4),
            round(float(np.percentile(reps, 97.5)), 4),
        ]

    out = ROOT / "outputs" / "dolphin_forecasting.json"
    out.write_text(
        json.dumps(
            {
                "study": "dolphin whistle forecasting — WITHIN-RECORDING scope "
                "(one 98-min recording; ID-block chronological-proxy partition; "
                "NO independent-session or caller claims)",
                "audit": audit,
                "release_enforcement": enforcement,
                "step2": {
                    "n_fit": len(tr),
                    "n_val": len(va),
                    "n_eval": len(ev),
                    "eval_blocks": len({r["group"] for r in ev}),
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
                    "secondary_ci": ci["delta_recent"],
                },
                "framing": "within-recording procedure test on this material "
                "only; no species comparison; magnitudes not biological constants",
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                "arms": {k: round(v["loss"], 4) for k, v in arms.items()},
                "delta_older": round(float(delta_older), 4),
                "ci": ci["delta_older"],
                "delta_recent": round(float(delta_recent), 4),
                "eval_blocks": len({r["group"] for r in ev}),
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
