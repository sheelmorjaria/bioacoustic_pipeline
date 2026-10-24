#!/usr/bin/env python3
"""Measurement manifest + 200 ms history-support audit (NO effect estimates).

STUDY (frozen spec follows in this artifact; execution order per review):
*Longer-history and context-dependent acoustic forecasting in bat C11/C12
bouts* — acoustic representation and 40 ms horizon FIXED; older history at
40/80/120/200 ms added under a common source-validity contract; five-model
nested ladder; ONE primary contrast
    Delta_context-history = L(M_context-recent) - L(M_context-older-history)
with supporting Delta_older-history = L(M_short) - L(M_shared-longer).

STEP 1 — MEASUREMENT MANIFEST (this script writes it): binds the study to
the ACTUAL extractor code (verified locally, not the docstrings):
  - physics (15-D), macrotexture (14-D), microtexture (15-D) ALL receive
    the SAME audio slice in extract(); a trajectory arises from repeated
    calls on successive windows. Macro/micro are FEATURE FAMILIES, NOT
    window scales — no presumed 'macrotexture = whole call' assignment;
  - shared helpers: spectral frames 25 ms window / 10 ms hop with FFT
    length 2*hop (approx 20 ms support), envelope 10 ms windows;
  - modulation estimator: 2–16 Hz band, returns (0,0) when no FFT bin
    falls in band — for short envelope segments the zeros are LACK OF
    SUPPORT, never measured absence of slow modulation;
  - whole-call Phi != aggregate of short-window Phi (nonlinear summaries,
    duration, ranges, onset intervals);
  - decision-time convention: s_t = Phi(x_[t-W,t)), W=40 ms; timestamp =
    LATEST source sample used; h >= W guarantees target-window
    non-overlap;
  - eligibility: B+W+h = 280 ms contiguous valid support (28 frames at
    10 ms hop); within-contiguous-span forecasting only — gaps are never
    compressed; the estimand is conditional on sufficiently long,
    continuously supported spans (no call endings/pauses/turn-taking
    claims).

STEP 2 — SUPPORT AUDIT (this script computes it; no models fitted):
per context / caller / recipient / recording group: eligible bouts,
eligible prediction positions, contiguous-span duration distribution,
exclusion reasons; dyadic cells frozen with the declared minimum.

Usage:
    python scripts/history_support_audit.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

WIN_MS, HOP_MS = 40.0, 10.0
B_MS = 200.0
H_MS = 40.0
REQUIRED_FRAMES = int((B_MS + WIN_MS + H_MS) / HOP_MS)  # 28
MIN_POSITIONS_PER_CELL = 200  # declared design setting for dyadic support
SEED = 0


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def measurement_manifest() -> dict:
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
        FEATURE_UNITS,
        supported_features,
    )

    return {
        "git_commit": _git_commit(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "representation": "supported 12-feature trajectory "
        f"({', '.join(supported_features())}) with declared units — "
        "UNCHANGED for this study",
        "units": dict(FEATURE_UNITS),
        "extractor_facts_verified_locally": [
            "physics (15-D), macrotexture (14-D), microtexture (15-D) all "
            "receive the SAME audio slice in extract(); one fingerprint "
            "per slice; trajectories arise from repeated calls",
            "macrotexture/microtexture are FEATURE FAMILIES, not window "
            "scales — no 'macrotexture=whole call, microtexture=within-call "
            "grammar' assignment is licensed",
            "shared helpers: stft_mag frame_ms=25, hop_ms=10, n_fft=2*hop "
            "(~20 ms spectral support); envelope hop_ms=win_ms=10",
            "modulation estimator: 2-16 Hz band; returns (0,0) when no bin "
            "falls in band — zeros from short segments are LACK OF "
            "SUPPORT, never measured absence of slow modulation",
            "whole-call Phi != aggregate of short-window Phi (nonlinear "
            "summaries, duration, ranges, onset intervals change with the "
            "slice); texture summaries are COMPRESSED temporal information "
            "(slopes/rise-fall/co-occurrence), not the full trajectory",
        ],
        "decision_time_convention": {
            "state": "s_t = Phi(x_[t-W,t)), W=40 ms",
            "target": "s_{t+h} = Phi(x_[t+h-W, t+h)), h=40 ms >= W => "
            "target window starts after the latest input sample",
            "timestamp": "latest source sample used by the predictor "
            "(never the nominal window centre)",
            "history_lags_ms": [40, 80, 120, 200],
            "status": "proposed design settings, not biologically validated timescales",
        },
        "eligibility": {
            "contiguous_required_frames": REQUIRED_FRAMES,
            "total_extent_ms": B_MS + WIN_MS + H_MS,
            "rule": "positions t where frames [t-20, t+4] (inclusive) are "
            "all valid within ONE contiguous supported span; gaps "
            "are never compressed; estimand is conditional on "
            "sufficiently long continuously supported spans",
        },
    }


def main() -> None:
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        contiguous_spans,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        filename_block_group,
        parse_annotations_csv,
    )

    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    meta = {a.file_name: a for a in annos}
    manifest = measurement_manifest()

    cells: dict = {}
    span_lengths = []
    n_bouts_total = n_bouts_eligible = 0
    n_positions = 0
    exclusion: dict = {}
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
            n_bouts_total += 1
            f = frames[offsets[i] : offsets[i + 1]]
            valid = ~np.all(f == 0, axis=1)
            spans = contiguous_spans(valid)
            best_positions = 0
            for sid in sorted(set(spans.tolist())):
                if sid < 0:
                    continue
                idx = np.where(spans == sid)[0]
                L = len(idx)
                span_lengths.append(L)
                if L >= REQUIRED_FRAMES:
                    best_positions += L - REQUIRED_FRAMES + 1
            if best_positions == 0:
                reason = (
                    "no contiguous valid span"
                    if not valid.any()
                    else f"longest span < {REQUIRED_FRAMES} frames"
                )
                exclusion[reason] = exclusion.get(reason, 0) + 1
                continue
            n_bouts_eligible += 1
            n_positions += best_positions
            key = (a.emitter, a.addressee, a.context)
            c = cells.setdefault(key, {"positions": 0, "bouts": 0, "groups": set()})
            c["positions"] += best_positions
            c["bouts"] += 1
            c["groups"].add(filename_block_group(nm))

    span_lengths = np.array(span_lengths)
    # frozen dyadic cells: caller-recipient pairs with BOTH contexts above
    # the declared minimum
    by_pair: dict = {}
    for (y, r, ctx), c in cells.items():
        by_pair.setdefault((y, r), {})[ctx] = c
    frozen = {}
    for (y, r), d in sorted(by_pair.items()):
        if (
            11 in d
            and 12 in d
            and all(d[c]["positions"] >= MIN_POSITIONS_PER_CELL for c in (11, 12))
        ):
            frozen[f"c{y}_r{r}"] = {
                "caller": y,
                "recipient": r,
                "c11_positions": d[11]["positions"],
                "c12_positions": d[12]["positions"],
                "c11_bouts": d[11]["bouts"],
                "c12_bouts": d[12]["bouts"],
                "n_groups": len(d[11]["groups"] | d[12]["groups"]),
            }

    report = {
        "study": "Longer-history and context-dependent acoustic forecasting "
        "(bat C11/C12): STEP 1+2 only — measurement manifest + support "
        "audit; NO models fitted, NO effect estimates",
        "measurement_manifest": manifest,
        "support_audit": {
            "bouts_total": n_bouts_total,
            "bouts_eligible": n_bouts_eligible,
            "eligible_positions_total": n_positions,
            "exclusions": exclusion,
            "contiguous_span_stats": {
                "n_spans": int(len(span_lengths)),
                "median_frames": float(np.median(span_lengths)) if len(span_lengths) else None,
                "p90_frames": float(np.percentile(span_lengths, 90)) if len(span_lengths) else None,
                "max_frames": int(span_lengths.max()) if len(span_lengths) else None,
                "n_spans_ge_required": int((span_lengths >= REQUIRED_FRAMES).sum()),
            },
            "per_context": {
                str(ctx): {
                    "positions": sum(
                        c["positions"] for (y, r, c2), c in cells.items() if c2 == ctx
                    ),
                    "bouts": sum(c["bouts"] for (y, r, c2), c in cells.items() if c2 == ctx),
                }
                for ctx in (11, 12)
            },
            "dyadic_cells_frozen": {
                "rule": f">= {MIN_POSITIONS_PER_CELL} eligible positions in "
                "BOTH contexts per caller-recipient pair (declared "
                "design setting, frozen before any effect estimate)",
                "n_pairs_supported": len(frozen),
                "pairs": frozen,
                "common_recipient_weighting": "equal across supported "
                "recipients within caller, identical in both contexts "
                "(frozen)",
            },
        },
        "frozen_study_spec": {
            "models": [
                "short b+Ax",
                "shared-longer b+Ax+Bu",
                "context-level +b_C c",
                "context-recent +c A_C x",
                "context-older-history +c B_C u",
            ],
            "primary_contrast": "Delta_context-history = "
            "L(M_context-recent) - L(M_context-older-history); POSITIVE = "
            "context-dependent older-history terms improve held-out "
            "prediction beyond context-level and recent-state terms",
            "supporting_contrast": "Delta_older-history = L(M_short) - L(M_shared-longer)",
            "evaluation_positions": "IDENTICAL positions across all arms "
            "(common eligible set); the short-history model's broader-"
            "cohort performance reported separately, never as comparator",
            "loss": "existing forecasting nRMSE implementation, exact "
            "normalisation and aggregation bound in the artifact; any "
            "caller/context-balanced weighting is a NEW aggregation, named",
            "scaling": "ONE common training-fitted scaler across contexts "
            "and arms; no per-bout normalisation (would use future "
            "material); no context-specific scaling",
            "uncertainty": "paired resampling of RECORDING GROUPS (whole "
            "groups, all bouts/positions), shared across arms",
            "population": "familiar callers, new recording groups; "
            "held-out-caller extension is a separate later test",
            "synthetic_fixtures_required_before_eval": [
                "shared dynamics + context means (must NOT read as older-history interaction)",
                "context-dependent recent-state effects + shared older "
                "history (recent-control must absorb them)",
                "known context-by-older-history effect (detection at intended support/noise)",
                "gaps/padding/unequal exposure (support + common positions + weighting)",
            ],
            "preregistration": "We will test whether behavioural context "
            "modifies the predictive contribution of older acoustic "
            "history beyond context-associated offsets and immediate-state "
            "effects. The acoustic representation and 40 ms forecast "
            "horizon remain fixed; observations up to 200 ms before the "
            "current state are added under a common source-validity "
            "contract. Models are compared on identical held-out positions "
            "within supported caller-recipient populations. A positive "
            "result is interpreted as incremental context-conditioned "
            "acoustic predictability under the tested model family — not "
            "grammar, reusable units, motor physiology or meaning.",
            "interpretation_limits": [
                "history-helps and order-matters are not interchangeable "
                "claims (order comparator deliberately out of scope)",
                "context-dependent linear mapping may approximate "
                "different regions of a shared nonlinear relationship",
                "label-conditioned prediction, not an audio-only context "
                "classifier, not future-behaviour forecasting",
            ],
        },
    }
    out = ROOT / "outputs" / "history_support_audit.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(
        json.dumps(
            {
                "bouts_total": n_bouts_total,
                "bouts_eligible": n_bouts_eligible,
                "eligible_positions": n_positions,
                "exclusions": exclusion,
                "spans_ge_required": report["support_audit"]["contiguous_span_stats"][
                    "n_spans_ge_required"
                ],
                "per_context": report["support_audit"]["per_context"],
                "n_dyadic_pairs_frozen": len(frozen),
            },
            indent=1,
            default=str,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
