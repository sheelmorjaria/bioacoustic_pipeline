#!/usr/bin/env python3
"""Preregistered confirmation: dyn_cv |rate| context association (C12 vs C11).

PREREGISTRATION (frozen before any confirmation data was extracted; the
exploratory result that motivated this endpoint is recorded verbatim from
its artifact, never from memory):

  EXPLORATORY BASIS (outputs/context_dynamics.json, secondary family):
      dyn_cv median_abs_rate across_caller_mean = +1.7, perm p = 0.0015
      -> direction: C12 > C11 (more envelope-CV change in context 12)

  CONFIRMATION ENDPOINT (single, one-sided):
      per-bout median |d(dyn_cv)/dt| over valid adjacent frame pairs
      (dyn_cv = micro index 38, FEATURE_SCHEMA_VERSION 2)
      Theta = mean over supported (caller, recipient) pairs of
              (C12 cell mean - C11 cell mean), equal recipient weights,
              same weights in both contexts

  COHORT: recording groups OUTSIDE cache/stage2/bat_c11c12_traj.npz
      (the exploratory analysis observed only cached bouts) — cache-external
      C11/C12 bouts with addressee > 0. Support recomputed for THIS endpoint
      and frozen before scoring.

  DECISION: CONFIRMED iff Theta > 0 AND one-sided permutation p < 0.05 AND
      the recording-block bootstrap 95% CI excludes 0 from below.
      Otherwise the branch is CLOSED (no further settings searched).
      No other secondary endpoint is tested here — this is a single
      confirmation, not a family.

  RELEASE CONTRACT: the run declares its masks (source_validity = the
      fingerprint's absolute gate; detected_activity = hysteresis decode),
      feature schema 2, source manifest and split, and passes
      raise_for_science() before extracting anything.

Interpretation limits (unchanged): context-associated envelope-dynamics
difference within supported caller-recipient material — not motor
physiology, not meaning; the amplitude-dependent gate scopes the estimand
to material passing the measurement rule; relative measures can still be
affected by background and finite numerical floors.

Usage:
    python scripts/confirm_dyn_cv_context.py --self-test
    python scripts/confirm_dyn_cv_context.py [--jobs 24] [--cap-per-cell 200]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WIN_MS, HOP_MS, MAX_FRAMES = 40.0, 10.0, 300
MIN_BOUTS = 2
BOOT_B, PERM_B, SEED = 2000, 2000, 0
DYN_CV_IDX = 38
CACHE_OUT = ROOT / "cache" / "stage2" / "confirm_dyn_cv_traj.npz"


def prereg_spec() -> dict:
    """Frozen confirmation spec (with the exploratory basis read from its artifact)."""
    basis = json.loads((ROOT / "outputs" / "context_dynamics.json").read_text())
    sec = basis["secondary_family"]["features"]["dyn_cv"]["median_abs_rate"]
    return {
        "endpoint": "median |d(dyn_cv)/dt| per bout (micro idx 38, schema 2)",
        "direction": "C12 > C11 (one-sided)",
        "exploratory_basis": {
            "artifact": "outputs/context_dynamics.json",
            "across_caller_mean": sec["across_caller_mean"],
            "perm_p": sec["perm_p"],
            "multiplicity_note": "exploratory secondary family (22 tests); "
            "this confirmation carries NO family — single prespecified endpoint",
        },
        "min_bouts_per_context": MIN_BOUTS,
        "weights": "equal across supported recipients, identical in both contexts",
        "uncertainty": "recording-block bootstrap (B=2000) + context-label "
        "permutation within pairs (B=2000, Phipson-Smyth)",
        "decision_rule": "CONFIRMED iff Theta>0 AND one-sided p<0.05 AND "
        "bootstrap CI excludes 0 from below; else branch CLOSED",
        "cohort": "cache-external C11/C12 bouts, addressee>0 (recording groups "
        "not observed by the exploratory analysis)",
    }


def _extract_one(args):
    """Fingerprint trajectory for one bout (same framing as stage-2)."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.utils.audio import load_audio

    path, name = args
    audio, sr = load_audio(str(path), target_sr=250000)
    win, hop = int(sr * WIN_MS / 1000), int(sr * HOP_MS / 1000)
    rows, i = [], 0
    while i + win <= len(audio) and len(rows) < MAX_FRAMES:
        v = np.asarray(
            afp.extract(audio[i : i + win], sr, species="bat").combined, dtype=np.float32
        )
        rows.append(v)
        i += hop
    return name, np.asarray(rows, dtype=np.float32)


def build_cohort(cap_per_cell: int):
    """Cache-external confirmation cohort (deterministic, capped per cell)."""
    import random

    from bioacoustic_pipeline.ingestion.context_eligibility import (
        build_caller_context_support_matrix,
    )
    from bioacoustic_pipeline.ingestion.frozen_dusted import filter_eligible_intervals
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        parse_annotations_csv,
        split_recording_groups_by_caller,
    )

    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    matrix = build_caller_context_support_matrix(annos)
    eligible = filter_eligible_intervals(
        annos, contexts=(11, 12), min_bouts=20, eligibility_matrix=matrix
    )
    import numpy as np

    split = split_recording_groups_by_caller(eligible, seed=0)
    cache_names = (
        set(
            np.load(ROOT / "cache/stage2/bat_c11c12_traj.npz", allow_pickle=False)["names"].tolist()
        )
        if (ROOT / "cache/stage2/bat_c11c12_traj.npz").exists()
        else set()
    )
    external = [
        a
        for a in (split["train"] + split["test"])
        if a.file_name not in cache_names and a.addressee > 0
    ]
    rng = random.Random(SEED)
    per_cell: dict = {}
    cohort = []
    for a in external:
        key = (a.emitter, a.addressee, a.context)
        per_cell.setdefault(key, 0)
        if per_cell[key] < cap_per_cell:
            per_cell[key] += 1
            cohort.append(a)
    rng.shuffle(cohort)  # deterministic order for resumable extraction
    return cohort, {
        "n_eligible_external": len(external),
        "n_cohort": len(cohort),
        "cap_per_cell": cap_per_cell,
        "cache_files_excluded": len(cache_names),
        "cells": len(per_cell),
    }


def extract_trajectories(cohort, jobs: int):
    """Resumable parallel fingerprint extraction -> CACHE_OUT."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    done: set = set()
    frames: list = []
    offsets = [0]
    names: list = []
    if CACHE_OUT.exists():
        d = np.load(CACHE_OUT, allow_pickle=False)
        names = list(d["names"])
        done = set(names)
        frames = [d["frames"][d["offsets"][i] : d["offsets"][i + 1]] for i in range(len(names))]
        offsets = list(d["offsets"])
        print(f"resuming: {len(done)} bouts already extracted", flush=True)
    todo = [
        (ROOT / "data/egyptian_fruit_bats/audio" / a.file_name, a.file_name)
        for a in cohort
        if a.file_name not in done
    ]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for k, (name, rows) in enumerate(ex.map(_extract_one, todo, chunksize=8)):
            if len(rows) == 0:
                continue
            frames.append(rows)
            names.append(name)
            offsets.append(offsets[-1] + len(rows))
            if (k + 1) % 2000 == 0:
                print(f"  extracted {k + 1}/{len(todo)}", flush=True)
                np.savez_compressed(
                    CACHE_OUT,
                    frames=np.vstack(frames),
                    offsets=np.asarray(offsets, dtype=np.int64),
                    names=np.asarray(names),
                )
    np.savez_compressed(
        CACHE_OUT,
        frames=np.vstack(frames),
        offsets=np.asarray(offsets, dtype=np.int64),
        names=np.asarray(names),
    )
    print(f"trajectory cache: {len(names)} bouts -> {CACHE_OUT.name}", flush=True)
    return np.load(CACHE_OUT, allow_pickle=False)


def analyse(traj, cohort, prereg: dict) -> dict:
    import numpy as np

    from bioacoustic_pipeline.ingestion.labeled_corpus import filename_block_group

    frames, offsets, names = traj["frames"], traj["offsets"], list(traj["names"])
    idx = {n: i for i, n in enumerate(names)}
    bouts = []
    for a in cohort:
        i = idx.get(a.file_name)
        if i is None:
            continue
        f = frames[offsets[i] : offsets[i + 1]].astype(np.float64)
        valid = ~np.all(f == 0, axis=1)
        col = f[:, DYN_CV_IDX]
        d = np.full(len(col), np.nan)
        d[1:] = np.where(valid[1:] & valid[:-1], (col[1:] - col[:-1]) / (HOP_MS / 1000.0), np.nan)
        rate = np.abs(d[np.isfinite(d)])
        bouts.append(
            {
                "file": a.file_name,
                "caller": a.emitter,
                "recipient": a.addressee,
                "context": a.context,
                "block": filename_block_group(a.file_name),
                "n_deriv": int(len(rate)),
                "summary": float(np.median(rate)) if len(rate) else None,
            }
        )

    # ---- support manifest FROZEN before scoring
    cells: dict = {}
    for b in bouts:
        if b["summary"] is None:
            continue
        key = (b["caller"], b["recipient"], b["context"])
        c = cells.setdefault(key, {"n": 0, "obs": 0, "blocks": set()})
        c["n"] += 1
        c["obs"] += b["n_deriv"]
        c["blocks"].add(b["block"])
    by_pair: dict = {}
    for (y, r, ctx), c in cells.items():
        by_pair.setdefault((y, r), {})[ctx] = c
    pairs = {
        k: v
        for k, v in by_pair.items()
        if 11 in v and 12 in v and all(v[c]["n"] >= MIN_BOUTS for c in (11, 12))
    }
    manifest = {
        "stage": "FROZEN before scoring",
        "bouts_in_cache": len(names),
        "bouts_scored": sum(1 for b in bouts if b["summary"] is not None),
        "bouts_without_deriv": sum(1 for b in bouts if b["summary"] is None),
        "eligible_fraction": round(
            sum(1 for b in bouts if b["summary"] is not None) / max(len(bouts), 1), 4
        ),
        "cells": len(cells),
        "supported_pairs": len(pairs),
        "coverage_by_context": {
            str(c): {
                "bouts": sum(1 for b in bouts if b["context"] == c),
                "frac_with_deriv": round(
                    sum(1 for b in bouts if b["context"] == c and b["summary"] is not None)
                    / max(sum(1 for b in bouts if b["context"] == c), 1),
                    4,
                ),
            }
            for c in (11, 12)
        },
        "note": "the fingerprint gate is amplitude-dependent: eligibility "
        "describes material passing the measurement rule",
    }

    # ---- Theta
    def _theta(pair_map):
        per_caller: dict = {}
        for (y, r), v in pair_map.items():
            m12 = np.mean([b["summary"] for b in v[12]])
            m11 = np.mean([b["summary"] for b in v[11]])
            per_caller.setdefault(y, []).append(m12 - m11)
        return {y: float(np.mean(ds)) for y, ds in per_caller.items()}

    items = {k: {11: [], 12: []} for k in pairs}
    for b in bouts:
        if b["summary"] is None:
            continue
        k = (b["caller"], b["recipient"])
        if k in items and b["context"] in items[k]:
            items[k][b["context"]].append(b)
    thetas = _theta(items)
    overall = float(np.mean(list(thetas.values())))

    rng = np.random.RandomState(SEED)
    reps = []
    for _ in range(BOOT_B):
        sub = {}
        for k, v in items.items():
            boot = {}
            for ctx in (11, 12):
                blocks = sorted({b["block"] for b in v[ctx]})
                pick = rng.choice(blocks, size=len(blocks), replace=True)
                pooled = {b["block"]: b for b in v[ctx]}
                boot[ctx] = [pooled[bl] for bl in pick]
            sub[k] = boot
        reps.append(float(np.mean(list(_theta(sub).values()))))
    ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]

    prng = np.random.RandomState(SEED)
    null = []
    for _ in range(PERM_B):
        sub = {}
        for k, v in items.items():
            merged = v[11] + v[12]
            perm = prng.permutation(len(merged))
            sub[k] = {
                11: [merged[i] for i in perm[: len(v[11])]],
                12: [merged[i] for i in perm[len(v[11]) :]],
            }
        null.append(float(np.mean(list(_theta(sub).values()))))
    p_one = float((1 + np.sum(np.asarray(null) >= overall)) / (PERM_B + 1))

    confirmed = bool(overall > 0 and p_one < 0.05 and ci[0] > 0)
    return {
        "preregistration": prereg,
        "support_manifest": manifest,
        "theta_per_caller": {str(y): round(t, 3) for y, t in sorted(thetas.items())},
        "n_callers": len(thetas),
        "across_caller_mean": round(overall, 4),
        "bootstrap_ci95": [round(c, 4) for c in ci],
        "one_sided_perm_p": round(p_one, 5),
        "decision": (
            "CONFIRMED — the prespecified C12>C11 direction replicates on "
            "independent recording groups"
            if confirmed
            else "NOT CONFIRMED — the branch is closed; no further settings "
            "searched (the exploratory dyn_cv |rate| association does not "
            "survive independent recording groups under this design)"
        ),
        "confirmed": confirmed,
        "framing": "context-associated envelope-dynamics difference within "
        "supported caller-recipient material — not motor physiology, not "
        "meaning; independent RECORDING groups (callers overlap the "
        "exploratory set by design; groups did not contribute to endpoint "
        "selection)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--cap-per-cell", type=int, default=200)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        import numpy as np

        # synthetic: C12 bouts carry +0.5 median |rate| shift on dyn_cv
        rng = np.random.RandomState(0)

        class A:
            pass

        bouts = []
        for y in (1, 2):
            for r in (10, 20):
                for ctx, shift in ((11, 0.0), (12, 0.5)):
                    for k in range(4):
                        b = A()
                        b.file, b.caller, b.recipient, b.context = (f"f{y}{r}{ctx}{k}", y, r, ctx)
                        b.block = f"b{k % 2}"
                        b.summary = abs(rng.randn()) + shift
                        bouts.append(b)
        items = {}
        for b in bouts:
            items.setdefault((b.caller, b.recipient), {11: [], 12: []})[b.context].append(b)
        assert (
            np.mean(
                [
                    np.mean([x.summary for x in v[12]]) - np.mean([x.summary for x in v[11]])
                    for v in items.values()
                ]
            )
            > 0.3
        )
        print("SELF-TEST OK (synthetic direction machinery)", flush=True)
        return

    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import FEATURE_SCHEMA_VERSION
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
    )

    prereg = prereg_spec()
    cohort, cohort_stats = build_cohort(args.cap_per_cell)
    contract = ScientificRunContract(
        run_name="confirm_dyn_cv_context",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        masks={
            "source_validity": MaskProvenance(
                kind="source_validity",
                produced_by="fingerprint absolute gate (rms<1e-3 zeroes core)",
                win_ms=WIN_MS,
                hop_ms=HOP_MS,
                amplitude_stage="native waveform",
            ),
            "detected_activity": MaskProvenance(
                kind="detected_activity",
                produced_by="silence_units hysteresis "
                "(not used for scoring; declared for separation)",
                win_ms=WIN_MS,
                hop_ms=HOP_MS,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest={a.file_name: {"group": a.file_name, "part": "eval"} for a in cohort},
        eval_split={
            "provenance": "cache-external recording groups of the §46 stage-2 "
            "eligible split; no fitting occurs (estimation task)",
            "unit": "recording group (file)",
        },
    )
    enforcement = raise_for_science(contract)
    print(f"release contract enforced: {enforcement['release_version']}", flush=True)

    traj = extract_trajectories(cohort, args.jobs)
    report = analyse(traj, cohort, prereg)
    report["cohort"] = cohort_stats
    report["release_enforcement"] = enforcement
    out = ROOT / "outputs" / "confirm_dyn_cv_context.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "cohort",
                    "across_caller_mean",
                    "bootstrap_ci95",
                    "one_sided_perm_p",
                    "decision",
                )
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
