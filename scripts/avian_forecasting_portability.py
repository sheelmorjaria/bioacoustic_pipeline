#!/usr/bin/env python3
"""Avian forecasting portability: pipit_id and littleowl_id (frozen procedure).

Same contract as the chiffchaff study (scripts/chiffchaff_forecasting_
portability.py): support-only audit first (terminal-insufficient
acceptable), then five arms (training-mean / persistence / state /
two-state / +older lags 40-200 ms), 40 ms window / 10 ms hop / 40 ms
horizon, species-specific scaler + ridge refit, alpha on train-side
validation, identical eligible positions, feature-contract gate at
fitting and scoring, recording-cluster paired bootstrap, caller-balanced
sensitivity named separately.

DELIBERATE BOUNDARIES TESTED (per review, these are not mere sample
count): pipit = song in a second songbird regime with identity partly
from song structure (provenance retained); littleowl = individual
territorial CALLS (non-song vocalisation class, sparser structure).
FRAMING: procedure portability; no cross-species memory claims; no
inference from single-corpus signs.

Usage:
    python scripts/avian_forecasting_portability.py --species pipit
    python scripts/avian_forecasting_portability.py --species littleowl

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SR = 44100
REQUIRED_FRAMES = 28
TRAIN_FRAC = 0.7
VAL_FRAC = 0.25
BOOT_B = 2000
SEED = 0
MIN_TOTAL_POSITIONS = 2000
MIN_GROUPS = 8

SPECIES = {
    "pipit": {
        "dir": "pipit_id",
        "wav_glob": "wav/pipit-fg/*.wav",
        "group_re": r"(pipit\d{4}fg_(?:train|test)_\d+)_[\d_]+\.wav",
        "provenance": "identity from colour rings OR song structure "
        "(provenance caveat retained in caller-level "
        "reads)",
    },
    "littleowl": {
        "dir": "littleowl_id",
        "wav_glob": "wav/littleowl-fg/*.wav",
        "group_re": r"(littleowl\d{4}fg_(?:train|test)_[^_]+)_\d+\.wav",
        "provenance": "identity partly inferred from territory "
        "and FM patterns (most males unringed)",
    },
}


def _extract(job):
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.utils.audio import load_audio

    path, name, species = job
    audio, sr = load_audio(str(path), target_sr=SR)
    win, hop = int(sr * 0.04), int(sr * 0.01)
    rows, i = [], 0
    while i * hop + win <= len(audio) and len(rows) < 600:
        rows.append(
            np.asarray(
                afp.extract(audio[i * hop : i * hop + win], sr, species=species).combined,
                dtype=np.float64,
            )
        )
        i += 1
    if len(rows) < REQUIRED_FRAMES:
        return None
    return name, np.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--species", required=True, choices=sorted(SPECIES))
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

    sp = SPECIES[args.species]
    raw = ROOT / "data" / "alp" / sp["dir"] / "v0.1.0" / "raw"
    wavs = sorted((raw / "wav" / f"{args.species}-fg").glob("*.wav"))
    print(f"[{args.species}] fg WAVs: {len(wavs)} | provenance: {sp['provenance']}", flush=True)

    jobs = [(p, p.name, args.species) for p in wavs]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = [r for r in ex.map(_extract, jobs, chunksize=8) if r]
    fidx = [_NAME_TO_IDX[n] for n in supported_features()]
    gre = re.compile(sp["group_re"])

    clips = []
    n_no_support = 0
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
        m = gre.search(name)
        group = m.group(1) if m else Path(name).stem
        if pos:
            clips.append({"name": name, "group": group, "frames": f, "valid": valid, "pos": pos})
        else:
            n_no_support += 1
    positions_total = sum(len(c["pos"]) for c in clips)
    groups = sorted({c["group"] for c in clips})
    print(
        json.dumps(
            {
                "clips_extracted": len(results),
                "no_support": n_no_support,
                "with_positions": len(clips),
                "positions": positions_total,
                "groups": len(groups),
            },
            indent=1,
        ),
        flush=True,
    )

    audit = {
        "clips_extracted": len(results),
        "no_support": n_no_support,
        "with_positions": len(clips),
        "positions": positions_total,
        "groups": len(groups),
        "provenance": sp["provenance"],
        "sufficiency_bars": {"positions": MIN_TOTAL_POSITIONS, "groups": MIN_GROUPS},
    }
    if positions_total < MIN_TOTAL_POSITIONS or len(groups) < MIN_GROUPS:
        out = ROOT / "outputs" / f"{args.species}_forecasting_portability.json"
        out.write_text(
            json.dumps(
                {
                    "study": f"{args.species} forecasting portability — audit only",
                    "audit": audit,
                    "outcome": "INSUFFICIENT SUPPORT — terminal; no models fitted",
                },
                indent=1,
            )
        )
        print("INSUFFICIENT SUPPORT — terminal", flush=True)
        return

    assert_feature_contract(clips[0]["frames"], context=f"{args.species} fitting")
    rows = []
    for c in clips:
        track = {"z": c["frames"], "context": 11}
        for t in c["pos"]:
            x, u, y, _ = _design(track, t)
            rows.append({"x": x, "u": u, "y": y, "group": c["group"], "clip": c["name"]})
    rec_ids = sorted({r["group"] for r in rows})
    fit_recs = {
        g
        for g in rec_ids
        if int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    rng = np.random.RandomState(SEED)
    fit_list = sorted(fit_recs)
    rng.shuffle(fit_list)
    val_recs = set(fit_list[: max(1, int(len(fit_list) * VAL_FRAC))])
    fit_recs = set(fit_list) - val_recs
    tr = [r for r in rows if r["group"] in fit_recs]
    va = [r for r in rows if r["group"] in val_recs]
    ev = [r for r in rows if r["group"] not in fit_recs | val_recs]

    files = sorted({c["name"] for c in clips})

    def _grp(n):
        m = gre.search(n)
        return m.group(1) if m else n

    part_by_file = {}
    for r in rows:
        part_by_file.setdefault(
            r["clip"],
            "fit" if r["group"] in fit_recs else ("val" if r["group"] in val_recs else "eval"),
        )
    contract = ScientificRunContract(
        run_name=f"{args.species}_forecasting_portability",
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
                produced_by=f"contiguous spans >= {REQUIRED_FRAMES} frames",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(
            files, [part_by_file[f] for f in files], [_grp(f) for f in files]
        ),
        eval_split={
            "provenance": "recording-group hash split 70/30 (frozen "
            "procedure); train-side VAL 25% for alpha",
            "unit": "recording group",
        },
    )
    enforcement = raise_for_science(contract)

    Ytr = np.stack([r["y"] for r in tr])
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-9
    Xtr_raw = np.stack([r["x"] for r in tr])
    Utr_raw = np.stack([r["u"] for r in tr])
    mu_x, sd_x = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-9
    mu_u, sd_u = Utr_raw.mean(0), Utr_raw.std(0) + 1e-9
    ev_names = {r["clip"] for r in ev}
    for c in clips:
        if c["name"] in ev_names:
            assert_feature_contract(c["frames"], context=f"{args.species} scoring")
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

    out = ROOT / "outputs" / f"{args.species}_forecasting_portability.json"
    out.write_text(
        json.dumps(
            {
                "study": f"{args.species} forecasting portability (frozen chiffchaff "
                "procedure; deliberate boundary: "
                + (
                    "second songbird song regime, identity partly song-structure-based"
                    if args.species == "pipit"
                    else "individual territorial CALLS (non-song class)"
                ),
                "audit": audit,
                "release_enforcement": enforcement,
                "step2": {
                    "n_fit": len(tr),
                    "n_val": len(va),
                    "n_eval": len(ev),
                    "eval_groups": len({r["group"] for r in ev}),
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
                "framing": "procedure portability; no cross-species memory claims; "
                "no inference from single-corpus signs",
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
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
