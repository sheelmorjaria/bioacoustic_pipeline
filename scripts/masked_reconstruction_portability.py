#!/usr/bin/env python3
"""Masked-reconstruction PORTABILITY: Bengalese (primary) + chiffchaff.

Frozen from the bat study — identical comparison, mask length, feature
contract, split rules, model family and loss; NO per-corpus tuning:

    arms (identical masked positions): boundary blend | forward | backward
    | bidirectional | bidirectional + G   [+ context where annotated]
    primary: Delta_whole-call = L(blend) - L(bidir+G)
    MASK = 70 ms central (7 frames at 10 ms hop)
    MIN_SPAN = 1.2 s contiguous supported

CORPUS ROLES (review-chosen): Bengalese = PRIMARY portability test
(natural syllable annotations; more discrete internal structure — the
plausible regime for whole-call help; descriptive stratification of
whether the mask sits WHOLLY INSIDE one annotated syllable or STRADDLES
a syllable boundary — descriptive only, never an endpoint change).
Chiffchaff = contrasting continuously-voiced song regime. STOP after
these two; marmoset (sparse spans), little owl (few groups) and dolphin
(within-recording) deliberately not run.

OUTCOME TABLE (frozen): blend wins -> local smoothness dominates across
regimes; bidir>blend but +G does not -> longer local context matters,
whole-event summary adds little; +G>bidir -> whole-vocalization state
constrains the missing interval; context beyond G -> incremental
context information about internal realisation. No semantic claims in
any direction.

Usage:
    python scripts/masked_reconstruction_portability.py --corpus bengalese
    python scripts/masked_reconstruction_portability.py --corpus chiffchaff

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

from masked_reconstruction import (
    ALPHAS,
    BOOT_B,
    MASK_FRAMES,
    MIN_SPAN_FRAMES,
    SEED,
    VAL_FRAC,
    _arm_matrix,
)

TRAIN_FRAC = 0.7


def _load_bengalese():
    """Long dev spans + syllable annotations (for the stratification)."""
    import numpy as np

    import bengalese_annotation_calibration as bac
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        contiguous_spans,
    )

    spans = []
    for bird in (0, 1, 2):
        dev = bac.load_bird_dev(bird, 90)
        if not dev:
            continue
        for b in dev:
            valid = b["valid"]
            for sid in sorted(set(contiguous_spans(valid).tolist())):
                if sid < 0:
                    continue
                idx = np.where(contiguous_spans(valid) == sid)[0]
                if len(idx) < MIN_SPAN_FRAMES:
                    continue
                # map local idx -> annotation frames (ann frames are
                # absolute in the bout; our span is a subrange)
                lo, hi = int(idx[0]), int(idx[-1]) + 1
                ann = [(a["label"], a["s"], a["e"]) for a in b["ann"]]
                spans.append(
                    {
                        "id": f"{b['wav']}#{lo}",
                        "group": b["wav"],
                        "z": b["track"][lo:hi],
                        "ann": ann,
                        "has_ctx": False,
                        "ctx": None,
                    }
                )
    return spans


def _load_chiffchaff():
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        contiguous_spans,
        supported_features,
    )

    from chiffchaff_forecasting_portability import _extract, _recording_id

    raw = ROOT / "data" / "alp" / "chiffchaff_id" / "v0.1.0" / "raw"
    wavs = sorted((raw / "wav" / "chiffchaff-fg").glob("*.wav"))
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(p, p.name) for p in wavs]
    with ProcessPoolExecutor(max_workers=24) as ex:
        results = [r for r in ex.map(_extract, jobs, chunksize=8) if r]
    fidx = [_NAME_TO_IDX[n] for n in supported_features()]
    spans = []
    for name, f44 in results:
        f = f44[:, fidx]
        valid = ~np.all(f == 0, axis=1)
        for sid in sorted(set(contiguous_spans(valid).tolist())):
            if sid < 0:
                continue
            idx = np.where(contiguous_spans(valid) == sid)[0]
            if len(idx) < MIN_SPAN_FRAMES:
                continue
            lo = int(idx[0])
            spans.append(
                {
                    "id": f"{name}#{lo}",
                    "group": _recording_id(name),
                    "z": f[idx],
                    "ann": [],
                    "has_ctx": False,
                    "ctx": None,
                }
            )
    return spans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, choices=("bengalese", "chiffchaff"))
    args = ap.parse_args()

    import numpy as np

    from acoustic_dynamics_report import ridge_fit, ridge_predict
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        supported_features,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        assert_feature_contract,
        raise_for_science,
        source_manifest_from,
    )

    feats = supported_features()
    spans = _load_bengalese() if args.corpus == "bengalese" else _load_chiffchaff()
    print(f"[{args.corpus}] long spans: {len(spans)} (>= {MIN_SPAN_FRAMES} frames)", flush=True)
    if len(spans) < 12:
        print("INSUFFICIENT LONG SPANS — terminal", flush=True)
        return
    assert_feature_contract(spans[0]["z"], context=f"{args.corpus} reconstruction")

    rows = []
    straddle = {"within": 0, "straddle": 0}
    for sp in spans:
        T = len(sp["z"])
        a0 = T // 2 - MASK_FRAMES // 2
        b0 = a0 + MASK_FRAMES
        if a0 < 2 or b0 + 2 >= T:
            continue
        z = sp["z"]
        rest = np.vstack([z[:a0], z[b0:]])
        G = np.concatenate(
            [
                np.median(rest, axis=0),
                np.percentile(rest, 75, axis=0) - np.percentile(rest, 25, axis=0),
                np.percentile(rest, 90, axis=0) - np.percentile(rest, 10, axis=0),
            ]
        )
        # Bengalese descriptive stratification (never an endpoint change):
        # does [a0, b0) sit wholly inside one annotated syllable?
        strat = None
        if sp["ann"]:
            inside = any(s <= a0 and b0 <= e for _, s, e in sp["ann"])
            crossing = any(s < b0 and e > a0 for _, s, e in sp["ann"])
            strat = (
                "within"
                if inside and not [1 for _, s, e in sp["ann"] if s > a0 and s < b0]
                else ("straddle" if crossing else "between")
            )
            if strat in straddle:
                straddle[strat] += 1
        rows.append(
            {
                "fa": np.concatenate([z[a0 - 1], z[a0 - 2]]),
                "ba": np.concatenate([z[b0], z[b0 + 1]]),
                "G": G,
                "C": 1.0,
                "Y": z[a0:b0],
                "span": sp["id"],
                "block": sp["group"],
                "strat": strat,
            }
        )
    print(f"masked rows: {len(rows)} | stratification: {straddle}", flush=True)

    blocks = sorted({r["block"] for r in rows})
    fit_blocks = {
        b
        for b in blocks
        if int(hashlib.sha1(b.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    rng2 = np.random.RandomState(SEED)
    fit_list = sorted(fit_blocks)
    rng2.shuffle(fit_list)
    val_blocks = set(fit_list[: max(1, int(len(fit_list) * VAL_FRAC))])
    fit_blocks = set(fit_list) - val_blocks
    tr = [r for r in rows if r["block"] in fit_blocks]
    va = [r for r in rows if r["block"] in val_blocks]
    ev = [r for r in rows if r["block"] not in fit_blocks | val_blocks]

    ids = sorted({r["span"] for r in rows})
    part = {
        r["span"]: "fit"
        if r["block"] in fit_blocks
        else ("val" if r["block"] in val_blocks else "eval")
        for r in rows
    }
    sorted({r["block"] for r in rows})
    contract = ScientificRunContract(
        run_name=f"masked_reconstruction_portability_{args.corpus}",
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
                produced_by=f"contiguous spans >= {MIN_SPAN_FRAMES} frames",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(ids, [part[i] for i in ids], ids),
        eval_split={
            "provenance": "recording-group hash split 70/30; VAL 25%",
            "unit": "recording group (bout/file)",
        },
    )
    enforcement = raise_for_science(contract)

    def _B(rs, key):
        return np.stack([r[key] for r in rs])

    stats = {}
    for key in ("fa", "ba", "G"):
        Xf = _B(tr, key)
        stats[key] = (Xf.mean(0), Xf.std(0) + 1e-9)
    Yall = np.concatenate([r["Y"] for r in tr])
    mu_y, sd_y = Yall.mean(0), Yall.std(0) + 1e-9

    def _mats(rs):
        fa = (_B(rs, "fa") - stats["fa"][0]) / stats["fa"][1]
        ba = (_B(rs, "ba") - stats["ba"][0]) / stats["ba"][1]
        G = (_B(rs, "G") - stats["G"][0]) / stats["G"][1]
        C = np.array([r["C"] for r in rs])
        Y = ((_B(rs, "Y") - mu_y) / sd_y).reshape(len(rs), -1)
        return fa, ba, G, C, Y

    fa_tr, ba_tr, G_tr, C_tr, Y_tr = _mats(tr)
    fa_va, ba_va, G_va, C_va, Y_va = _mats(va)
    fa_ev, ba_ev, G_ev, C_ev, Y_ev = _mats(ev)

    def _loss(resid):
        r3 = resid.reshape(len(resid), MASK_FRAMES, len(feats))
        return float(np.sqrt((r3**2).mean(axis=0)).mean())

    arms_out, resid = {}, {}
    fa_raw = _B(ev, "fa")[:, :12]
    ba_raw = _B(ev, "ba")[:, :12]
    blend = np.zeros((len(ev), MASK_FRAMES, len(feats)))
    for k in range(MASK_FRAMES):
        w = (k + 1) / (MASK_FRAMES + 1)
        est = (1 - w) * fa_raw + w * ba_raw
        blend[:, k, :] = (est - mu_y) / sd_y
    resid["local-interp"] = blend.reshape(len(ev), -1) - Y_ev
    arms_out["local-interp"] = {"loss": _loss(resid["local-interp"])}

    for arm in ("forward", "backward", "bidir", "whole-call", "context"):
        A_tr = _arm_matrix(arm, fa_tr, ba_tr, G_tr, C_tr)
        A_va = _arm_matrix(arm, fa_va, ba_va, G_va, C_va)
        A_ev = _arm_matrix(arm, fa_ev, ba_ev, G_ev, C_ev)
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(A_tr, Y_tr, alpha)
            lv = _loss(ridge_predict(w, b, A_va) - Y_va)
            if lv < best[1]:
                best = (alpha, lv)
        w, b = ridge_fit(A_tr, Y_tr, best[0])
        r_ = ridge_predict(w, b, A_ev) - Y_ev
        resid[arm] = r_
        arms_out[arm] = {"alpha": best[0], "loss": _loss(r_)}

    delta_whole = arms_out["local-interp"]["loss"] - arms_out["whole-call"]["loss"]
    delta_bidir = arms_out["local-interp"]["loss"] - arms_out["bidir"]["loss"]
    delta_G = arms_out["bidir"]["loss"] - arms_out["whole-call"]["loss"]

    by_b: dict = {}
    for i, r in enumerate(ev):
        by_b.setdefault(r["block"], []).append(i)
    keys = sorted(by_b)
    brng = np.random.RandomState(SEED)
    cis = {}
    for name, (ra, rb) in {
        "delta_whole-call": ("local-interp", "whole-call"),
        "delta_bidir": ("local-interp", "bidir"),
        "delta_G_beyond_bidir": ("bidir", "whole-call"),
    }.items():
        reps = []
        for _ in range(BOOT_B):
            pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
            idx = np.concatenate([by_b[g] for g in pick])
            reps.append(_loss(resid[ra][idx]) - _loss(resid[rb][idx]))
        cis[name] = [
            round(float(np.percentile(reps, 2.5)), 4),
            round(float(np.percentile(reps, 97.5)), 4),
        ]

    # descriptive stratified losses (Bengalese; never an endpoint)
    strat_losses = {}
    for st in ("within", "straddle", "between"):
        idx = [i for i, r in enumerate(ev) if r["strat"] == st]
        if len(idx) >= 3:
            strat_losses[st] = {
                "n": len(idx),
                "blend": round(_loss(resid["local-interp"][idx]), 4),
                "whole-call": round(_loss(resid["whole-call"][idx]), 4),
            }

    out = ROOT / "outputs" / f"masked_reconstruction_{args.corpus}.json"
    out.write_text(
        json.dumps(
            {
                "study": f"masked-reconstruction PORTABILITY — {args.corpus} "
                "(frozen bat comparison; 70 ms central mask; no per-corpus tuning)",
                "release_enforcement": enforcement,
                "arms": {
                    k: (
                        {
                            "loss": round(v["loss"], 4),
                            **({"alpha": v["alpha"]} if "alpha" in v else {}),
                        }
                    )
                    for k, v in arms_out.items()
                },
                "deltas": {
                    "delta_whole-call": round(float(delta_whole), 4),
                    "delta_bidir": round(float(delta_bidir), 4),
                    "delta_G_beyond_bidir": round(float(delta_G), 4),
                },
                "bootstrap_ci": cis,
                "stratification_descriptive": strat_losses or straddle,
                "counts": {
                    "n_spans": len(spans),
                    "n_rows": len(rows),
                    "n_fit": len(tr),
                    "n_val": len(va),
                    "n_eval": len(ev),
                    "eval_blocks": len(keys),
                },
                "outcome_table": "blend wins -> local smoothness dominates across "
                "regimes; bidir>blend, +G not -> longer local context matters, "
                "whole-event summary adds little; +G>bidir -> whole-voc state "
                "constrains the interval; no semantic claims",
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                "arms": {k: round(v["loss"], 4) for k, v in arms_out.items()},
                "deltas": {
                    "whole-call": round(float(delta_whole), 4),
                    "bidir": round(float(delta_bidir), 4),
                    "G_beyond_bidir": round(float(delta_G), 4),
                },
                "cis": cis,
                "strat": strat_losses or straddle,
                "n_eval": len(ev),
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
