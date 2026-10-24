#!/usr/bin/env python3
"""Bounded C11-C12 context-dynamics analysis (new acoustic endpoint).

Question: within the same caller and recipient comparison, does context
change the rate or direction of a selected acoustic feature?

Prespecified endpoint (chosen for acoustic interpretation, BEFORE inspecting
context effects):
  PRIMARY feature  : centroid (spectral-centre motion)
  PRIMARY summaries: (a) median SIGNED d(centroid)/dt  — direction of change
                     (b) median |d(centroid)/dt|       — amount of change
  SECONDARY family : the other 11 supported features, same two summaries,
                     reported as descriptive with explicit multiplicity.

Estimand per caller y over supported recipients R_y* (both contexts with
>= MIN_BOUTS bouts carrying >= 1 valid derivative):

    Theta_y = sum_{r in R_y*} w_yr (mean_b a_{y,r,C12} - mean_b a_{y,r,C11}),
    w_yr = 1/|R_y*|   (same weights in both contexts)

Uncertainty: recording-block-clustered bootstrap (blocks resampled within
each cell) for CIs; context-label permutation within (caller, recipient)
pairs for the two-sided p-value (Phipson-Smyth correction).

Order of operations (frozen): the support manifest is computed and written
FIRST;Theta evaluation reads only the manifest's supported cells. No cell is
filled with zero and no weight changes after effects are seen.

Data: cache/stage2/bat_c11c12_traj.npz (frozen §46 stage-2 trajectories, WIN
40 ms / HOP 10 ms; validity = the fingerprint's own gate). Derivative
support is RECOMPUTED here (adjacent valid pairs), never inherited from the
forecasting cohort or an earlier dyadic eligibility count. Framing: a new,
prospectively specified acoustic endpoint on the established corpus — not
independent-data replication, and NOT a motor-physiology or meaning claim.

Usage:
    python scripts/context_dynamics.py --self-test
    python scripts/context_dynamics.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MIN_BOUTS = 2  # per (caller, recipient, context): bouts with >=1 derivative
BOOT_B = 2000
PERM_B = 2000
SEED = 0
CENTROID_IDX = 5  # verified by test_layout_pinned_to_extractor


def _bout_summaries(sub, valid, dt: float):
    """Median signed / |.| derivative over valid adjacent pairs, per feature."""
    import numpy as np

    d = np.full_like(sub, np.nan)
    for j in range(sub.shape[1]):
        dj = np.full(len(sub), np.nan)
        dj[1:] = np.where(valid[1:] & valid[:-1], (sub[1:, j] - sub[:-1, j]) / dt, np.nan)
        d[:, j] = dj
    obs = np.isfinite(d)
    n_obs = int(obs.any(axis=1).sum())
    signed = np.where(obs.any(axis=0), np.nanmedian(np.where(obs, d, np.nan), axis=0), np.nan)
    absolute = np.where(
        obs.any(axis=0), np.nanmedian(np.where(obs, np.abs(d), np.nan), axis=0), np.nan
    )
    return signed, absolute, n_obs, d


def build_support_and_summaries(cache_path: Path):
    """Stage 1 (frozen first): per-bout derivative support + summaries."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
        _NAME_TO_IDX,
        supported_features,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        filename_block_group,
        parse_annotations_csv,
    )

    data = np.load(cache_path, allow_pickle=False)
    frames, offsets = data["frames"], data["offsets"]
    names, parts = list(data["names"]), data["parts"]
    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    meta = {a.file_name: a for a in annos}
    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]

    seen: dict = {}
    for i, nm in enumerate(names):
        seen.setdefault(nm, i)  # dedupe: first occurrence
    bouts = []
    for nm, i in seen.items():
        a = meta.get(nm)
        if a is None or a.context not in (11, 12) or a.addressee <= 0:
            continue
        f = frames[offsets[i] : offsets[i + 1]].astype(np.float64)
        valid = ~np.all(f == 0, axis=1)
        signed, absolute, n_obs, _ = _bout_summaries(f[:, fidx], valid, dt=0.01)
        bouts.append(
            {
                "file": nm,
                "caller": a.emitter,
                "recipient": a.addressee,
                "context": a.context,
                "block": filename_block_group(nm),
                "part": int(parts[i]),
                "n_frames": int(len(f)),
                "n_valid_frames": int(valid.sum()),
                "n_deriv_obs": n_obs,
                "signed": signed,
                "absolute": absolute,
            }
        )
    return bouts, feats, FEATURE_SCHEMA_VERSION


def freeze_manifest(bouts, feats) -> dict:
    """Support manifest — written BEFORE any Theta evaluation."""
    cells: dict = {}
    for b in bouts:
        key = (b["caller"], b["recipient"], b["context"])
        c = cells.setdefault(
            key, {"files": 0, "bouts_with_deriv": 0, "deriv_obs": 0, "blocks": set()}
        )
        c["files"] += 1
        c["bouts_with_deriv"] += 1 if b["n_deriv_obs"] > 0 else 0
        c["deriv_obs"] += b["n_deriv_obs"]
        c["blocks"].add(b["block"])

    manifest = {
        "stage": "FROZEN before Theta evaluation",
        "min_bouts_per_context": MIN_BOUTS,
        "cells": {
            f"c{y}_r{r}_ctx{ctx}": {
                "caller": y,
                "recipient": r,
                "context": ctx,
                "files": c["files"],
                "bouts_with_deriv": c["bouts_with_deriv"],
                "deriv_obs": c["deriv_obs"],
                "n_blocks": len(c["blocks"]),
            }
            for (y, r, ctx), c in sorted(cells.items())
        },
        "coverage_by_context": {},
    }
    for ctx in (11, 12):
        bs = [b for b in bouts if b["context"] == ctx]
        manifest["coverage_by_context"][str(ctx)] = {
            "bouts": len(bs),
            "frac_with_deriv": round(sum(b["n_deriv_obs"] > 0 for b in bs) / max(len(bs), 1), 4),
            "median_deriv_obs_per_bout": (
                float(__import__("numpy").median([b["n_deriv_obs"] for b in bs])) if bs else 0.0
            ),
            "note": "the fingerprint gate is amplitude-dependent: coverage "
            "describes the material that passes the measurement rule",
        }
    manifest["supported_caller_recipient_pairs"] = {}
    by_pair: dict = {}
    for (y, r, ctx), c in cells.items():
        by_pair.setdefault((y, r), {})[ctx] = c
    for (y, r), d in sorted(by_pair.items()):
        if 11 in d and 12 in d and all(d[c]["bouts_with_deriv"] >= MIN_BOUTS for c in (11, 12)):
            manifest["supported_caller_recipient_pairs"][f"c{y}_r{r}"] = {
                "caller": y,
                "recipient": r,
                "c11": {"bouts": d[11]["bouts_with_deriv"], "obs": d[11]["deriv_obs"]},
                "c12": {"bouts": d[12]["bouts_with_deriv"], "obs": d[12]["deriv_obs"]},
            }
    return manifest


def evaluate_theta(bouts, feats, *, summary: str, feat_idx: int):
    """Theta per caller + across-caller summary with block bootstrap + perm."""
    import numpy as np

    by_pair: dict = {}
    for b in bouts:
        if b["n_deriv_obs"] == 0:
            continue
        by_pair.setdefault((b["caller"], b["recipient"]), {11: [], 12: []})[b["context"]].append(b)

    supported = {
        k: v for k, v in by_pair.items() if len(v[11]) >= MIN_BOUTS and len(v[12]) >= MIN_BOUTS
    }
    if not supported:
        return {"note": "no supported caller-recipient pairs", "n_callers": 0}

    def _cell_mean(items):
        vals = [b[summary][feat_idx] for b in items]
        return float(np.mean(vals))

    def _theta(pairs_subset):
        per_caller: dict = {}
        for (y, r), v in pairs_subset.items():
            per_caller.setdefault(y, []).append(_cell_mean(v[12]) - _cell_mean(v[11]))
        return {y: float(np.mean(ds)) for y, ds in per_caller.items()}

    thetas = _theta(supported)
    overall = float(np.mean(list(thetas.values())))

    # recording-block bootstrap: resample blocks within each cell
    rng = np.random.RandomState(SEED)
    reps = []
    for _ in range(BOOT_B):
        sub = {}
        for k, v in supported.items():
            boot = {}
            for ctx in (11, 12):
                blocks = sorted({b["block"] for b in v[ctx]})
                pick = rng.choice(blocks, size=len(blocks), replace=True)
                pooled = {b["block"]: b for b in v[ctx]}
                items = [pooled[blk] for blk in pick]
                boot[ctx] = items
            sub[k] = boot
        reps.append(float(np.mean(list(_theta(sub).values()))))
    ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]

    # context-label permutation within supported pairs (label swap of bouts)
    prng = np.random.RandomState(SEED)
    null = []
    for _ in range(PERM_B):
        sub = {}
        for k, v in supported.items():
            merged = v[11] + v[12]
            perm = prng.permutation(len(merged))
            sub[k] = {
                11: [merged[i] for i in perm[: len(v[11])]],
                12: [merged[i] for i in perm[len(v[11]) :]],
            }
        null.append(float(np.mean(list(_theta(sub).values()))))
    p = float((1 + np.sum(np.abs(null) >= abs(overall))) / (PERM_B + 1))

    return {
        "n_supported_pairs": len(supported),
        "n_callers": len(thetas),
        "theta_per_caller": {str(y): round(t, 2) for y, t in sorted(thetas.items())},
        "across_caller_mean": round(overall, 4),
        "bootstrap_ci95": [round(c, 4) for c in ci],
        "perm_p": round(p, 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        import numpy as np

        rng = np.random.RandomState(0)
        n_feat = 12
        bouts = []
        for y in (1, 2):
            for r in (10, 20):
                for ctx, shift in ((11, 0.0), (12, 400.0)):
                    for k in range(4):
                        T = 40
                        sub = rng.randn(T, n_feat)
                        sub[:, CENTROID_IDX] += np.arange(T) * (shift + 100.0 * k)
                        valid = np.ones(T, dtype=bool)
                        signed, absolute, n_obs, _ = _bout_summaries(sub, valid, 0.01)
                        bouts.append(
                            {
                                "file": f"f{y}_{r}_{ctx}_{k}",
                                "caller": y,
                                "recipient": r,
                                "context": ctx,
                                "block": f"b{k % 2}",
                                "n_deriv_obs": n_obs,
                                "signed": signed,
                                "absolute": absolute,
                            }
                        )
        manifest = freeze_manifest(bouts, None)
        assert len(manifest["supported_caller_recipient_pairs"]) == 4
        res = evaluate_theta(bouts, None, summary="signed", feat_idx=CENTROID_IDX)
        assert res["across_caller_mean"] > 100.0, res  # C12 raises centroid rate
        assert res["perm_p"] <= 0.05, res
        print(f"SELF-TEST OK theta={res['across_caller_mean']} p={res['perm_p']}", flush=True)
        return

    import numpy as np

    cache = ROOT / "cache" / "stage2" / "bat_c11c12_traj.npz"
    bouts, feats, schema = build_support_and_summaries(cache)
    manifest = freeze_manifest(bouts, feats)
    manifest["feature_schema_version"] = schema
    manifest["data"] = {
        "cache": str(cache.relative_to(ROOT)),
        "bouts_used": len(bouts),
        "framing": "new prospectively specified acoustic endpoint on the "
        "established corpus (not independent replication)",
    }
    sup = ROOT / "outputs" / "context_dynamics_support.json"
    sup.write_text(json.dumps(manifest, indent=1))
    n_pairs = len(manifest["supported_caller_recipient_pairs"])
    print(
        f"support frozen: {len(bouts)} bouts, {len(manifest['cells'])} cells, "
        f"{n_pairs} supported caller-recipient pairs",
        flush=True,
    )
    print(f"coverage: {json.dumps(manifest['coverage_by_context'])}", flush=True)

    report = {
        "task": "context-associated acoustic dynamics (C12 - C11) within "
        "supported caller-recipient pairs",
        "estimand": "Theta_y = mean over supported recipients of (C12 mean - "
        "C11 mean) of the bout-level summary; equal recipient weights, same "
        "weights in both contexts",
        "primary": {
            "feature": "centroid",
            "units": "Hz/s",
            "median_signed_rate": evaluate_theta(
                bouts, feats, summary="signed", feat_idx=feats.index("centroid")
            ),
            "median_abs_rate": evaluate_theta(
                bouts, feats, summary="absolute", feat_idx=feats.index("centroid")
            ),
        },
        "secondary_family": {
            "note": "DESCRIPTIVE with explicit multiplicity (11 features x 2 "
            "summaries); no selection among them is used for any claim",
            "features": {
                name: {
                    "median_signed_rate": evaluate_theta(
                        bouts, feats, summary="signed", feat_idx=j
                    ),
                    "median_abs_rate": evaluate_theta(bouts, feats, summary="absolute", feat_idx=j),
                }
                for j, name in enumerate(feats)
                if name != "centroid"
            },
        },
        "interpretation_limits": [
            "context-associated differences in an acoustic-rate measurement "
            "within the supported caller-recipient population",
            "not a motor-physiology claim; not meaning",
            "not an attempt to reverse the frozen unit-presence or CPC-interaction nulls",
            "the validity gate is amplitude-dependent: the estimand concerns "
            "material that passes the measurement rule",
        ],
    }
    out = ROOT / "outputs" / "context_dynamics.json"
    out.write_text(json.dumps(report, indent=1))
    prim = report["primary"]
    print(f"PRIMARY centroid signed: {json.dumps(prim['median_signed_rate'])}", flush=True)
    print(f"PRIMARY centroid |rate|: {json.dumps(prim['median_abs_rate'])}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
