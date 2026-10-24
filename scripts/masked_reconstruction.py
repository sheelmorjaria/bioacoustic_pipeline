#!/usr/bin/env python3
"""Masked within-vocalization reconstruction (frozen spec, bat C11/C12).

QUESTION: does the REST of a vocalization constrain a hidden internal
interval beyond its immediate local neighbourhood?

    Delta_whole-call = L(local-only) - L(whole-voc context)   [primary]
    Delta_C          = L(whole-call)  - L(whole-call + context) [secondary]

DESIGN (prespecified): in long (>=1.2 s) contiguous supported bat spans,
mask a real central interval of 7 frames (70 ms). Arms on IDENTICAL
masked positions:
  A1 local-interp  — linear blend of the boundary frames (strong, simple)
  A2 forward-only  — causal ridge from [s_{a-1}, s_{a-2}]
  A3 backward-only — anti-causal ridge from [s_{b}, s_{b+1}]
  A4 bidirectional — [s_{a-1}, s_{a-2}, s_{b}, s_{b+1}]
  A5 + whole-call G — A4 + summaries of the UNMASKED rest (median, IQR,
      q10/q90 per feature over valid unmasked frames)
  A6 + context     — A5 + C11/C12 code
Provenance states enforced in the artifact: observed / interpolated /
invalid / missing are distinct; masked-reconstruction outputs are never
re-treated as observed acoustic evidence. G is computed from the
UNMASKED rest only (no leakage of the target). Scaler per-arm from FIT
material; ridge alpha on train-side validation; recording-block split;
block-cluster paired bootstrap.

OUTCOME TABLE (prespecified): positive Delta_whole-call => "other parts
of the same vocalization constrain the missing internal acoustics
beyond the immediate local neighbourhood" — vocalization-level
ORGANISATION, not words/syntax/meaning. Secondary Delta_C => context
adds information about internal realisation beyond the observed
vocalization itself. Neither is a semantic claim.

Usage:
    python scripts/masked_reconstruction.py

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

MASK_FRAMES = 7  # 70 ms central mask
MIN_SPAN_FRAMES = 120  # 1.2 s span requirement
TRAIN_FRAC = 0.7
VAL_FRAC = 0.25
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
BOOT_B = 2000
SEED = 0
ARMS = ("local-interp", "forward", "backward", "bidir", "whole-call", "context")


def _arm_matrix(arm, fa, ba, G, C):
    """fa = forward pair, ba = backward pair, G = rest summaries, C = +-1."""
    import numpy as np

    if arm == "local-interp":
        return None
    if arm == "forward":
        return fa
    if arm == "backward":
        return ba
    if arm == "bidir":
        return np.hstack([fa, ba])
    if arm == "whole-call":
        return np.hstack([fa, ba, G])
    if arm == "context":
        return np.hstack([fa, ba, G, C[:, None]])
    raise ValueError(arm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    import numpy as np

    from acoustic_dynamics_report import ridge_fit, ridge_predict
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
        _NAME_TO_IDX,
        contiguous_spans,
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
        source_manifest_from,
    )

    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    meta = {a.file_name: a for a in annos}
    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]

    spans = []
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
            if a.context not in (11, 12) or a.addressee <= 0:
                continue
            f = frames[offsets[i] : offsets[i + 1]][:, fidx].astype(np.float64)
            valid = ~np.all(f == 0, axis=1)
            for sid in sorted(set(contiguous_spans(valid).tolist())):
                if sid < 0:
                    continue
                idx = np.where(contiguous_spans(valid) == sid)[0]
                if len(idx) < MIN_SPAN_FRAMES:
                    continue
                spans.append(
                    {
                        "file": nm,
                        "ctx": a.context,
                        "block": filename_block_group(nm),
                        "z": f[idx],
                        "lo": int(idx[0]),
                    }
                )
    print(f"long spans: {len(spans)} (>= {MIN_SPAN_FRAMES} frames)", flush=True)
    if len(spans) < 12:
        out = ROOT / "outputs" / "masked_reconstruction.json"
        out.write_text(
            json.dumps({"outcome": "INSUFFICIENT LONG SPANS", "n_spans": len(spans)}, indent=1)
        )
        print("INSUFFICIENT LONG SPANS — terminal", flush=True)
        return
    assert_feature_contract(spans[0]["z"], context="reconstruction fitting")

    # ---- build masked rows: one per span (central mask, deterministic)
    rows = []
    for sp in spans:
        T = len(sp["z"])
        a0 = T // 2 - MASK_FRAMES // 2
        b0 = a0 + MASK_FRAMES
        if a0 < 2 or b0 + 2 >= T:
            continue
        z = sp["z"]
        # G from the UNMASKED rest only
        rest = np.vstack([z[:a0], z[b0:]])
        G = np.concatenate(
            [
                np.median(rest, axis=0),
                np.percentile(rest, 75, axis=0) - np.percentile(rest, 25, axis=0),
                np.percentile(rest, 90, axis=0) - np.percentile(rest, 10, axis=0),
            ]
        )
        rows.append(
            {
                "fa": np.concatenate([z[a0 - 1], z[a0 - 2]]),
                "ba": np.concatenate([z[b0], z[b0 + 1]]),
                "G": G,
                "C": 1.0 if sp["ctx"] == 12 else -1.0,
                "Y": z[a0:b0],  # (7, 12) target block
                "span": sp["file"],
                "block": sp["block"],
                "a0": a0,
            }
        )
    print(f"masked rows: {len(rows)} (1 per span, central 70 ms)", flush=True)

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

    files = sorted({r["span"] for r in rows})
    part = {}
    for r in rows:
        part.setdefault(
            r["span"],
            "fit" if r["block"] in fit_blocks else ("val" if r["block"] in val_blocks else "eval"),
        )
    contract = ScientificRunContract(
        run_name="masked_reconstruction",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
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
        source_manifest=source_manifest_from(
            files, [part[f] for f in files], [filename_block_group(f) for f in files]
        ),
        eval_split={
            "provenance": "recording-block hash split 70/30; VAL 25%",
            "unit": "recording block",
        },
    )
    enforcement = raise_for_science(contract)

    def _blocks(rs, key):
        return np.stack([r[key] for r in rs])

    # standardisation stats from FIT
    stats = {}
    for key in ("fa", "ba", "G"):
        Xf = _blocks(tr, key)
        stats[key] = (Xf.mean(0), Xf.std(0) + 1e-9)
    Yall = np.concatenate([r["Y"] for r in tr])
    mu_y, sd_y = Yall.mean(0), Yall.std(0) + 1e-9

    def _mats(rs):
        fa = (_blocks(rs, "fa") - stats["fa"][0]) / stats["fa"][1]
        ba = (_blocks(rs, "ba") - stats["ba"][0]) / stats["ba"][1]
        G = (_blocks(rs, "G") - stats["G"][0]) / stats["G"][1]
        C = np.array([r["C"] for r in rs])
        Y = (np.stack([r["Y"] for r in rs]) - mu_y) / sd_y  # (n, 7, 12)
        Yf = Y.reshape(len(rs), -1)  # flattened target
        return fa, ba, G, C, Yf

    fa_tr, ba_tr, G_tr, C_tr, Y_tr = _mats(tr)
    fa_va, ba_va, G_va, C_va, Y_va = _mats(va)
    fa_ev, ba_ev, G_ev, C_ev, Y_ev = _mats(ev)

    def _loss(resid):
        # per-position nRMSE averaged over the 7 masked frames x 12 features
        r3 = resid.reshape(len(resid), MASK_FRAMES, len(feats))
        return float(np.sqrt((r3**2).mean(axis=0)).mean())

    arms_out, resid = {}, {}
    # A1 local interpolation (no fit): linear blend of the boundary frames
    n_ev = len(ev)
    fa_raw = _blocks(ev, "fa")[:, :12]  # s_{a-1} (unnormalized)
    ba_raw = _blocks(ev, "ba")[:, :12]  # s_{b}
    blend = np.zeros((n_ev, MASK_FRAMES, len(feats)))
    for k in range(MASK_FRAMES):
        w = (k + 1) / (MASK_FRAMES + 1)
        est = (1 - w) * fa_raw + w * ba_raw
        blend[:, k, :] = (est - mu_y) / sd_y
    resid["local-interp"] = blend.reshape(n_ev, -1) - Y_ev
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
    delta_C = arms_out["whole-call"]["loss"] - arms_out["context"]["loss"]

    by_b: dict = {}
    for i, r in enumerate(ev):
        by_b.setdefault(r["block"], []).append(i)
    keys = sorted(by_b)
    brng = np.random.RandomState(SEED)
    cis = {}
    for name, (ra, rb) in {
        "delta_whole-call (local-interp - whole-call)": ("local-interp", "whole-call"),
        "delta_bidir (local-interp - bidir)": ("local-interp", "bidir"),
        "delta_C (whole-call - context)": ("whole-call", "context"),
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

    out = ROOT / "outputs" / "masked_reconstruction.json"
    out.write_text(
        json.dumps(
            {
                "study": "masked within-vocalization reconstruction (central 70 ms "
                f"in >=1.2 s supported spans; {len(rows)} spans)",
                "release_enforcement": enforcement,
                "provenance_states": "observed / interpolated / invalid / missing "
                "kept distinct; reconstructions never re-treated as observed",
                "G_definition": "median + IQR + (q90-q10) per feature over the "
                "UNMASKED rest (no target leakage)",
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
                    "delta_C": round(float(delta_C), 4),
                },
                "bootstrap_ci": cis,
                "counts": {
                    "n_spans": len(spans),
                    "n_rows": len(rows),
                    "n_fit": len(tr),
                    "n_val": len(va),
                    "n_eval": len(ev),
                    "eval_blocks": len(keys),
                },
                "outcome_table": "positive Delta_whole-call => other parts of the "
                "same vocalization constrain the missing internal acoustics beyond "
                "the immediate local neighbourhood — vocalisation-level "
                "ORGANISATION, not words/syntax/meaning; positive Delta_C => "
                "context adds information about internal realisation beyond the "
                "observed vocalization itself",
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
                    "C": round(float(delta_C), 4),
                },
                "cis": cis,
                "n_eval": len(ev),
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
