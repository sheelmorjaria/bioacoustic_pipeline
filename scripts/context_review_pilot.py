#!/usr/bin/env python3
"""Assisted context-review pilot: auditable annotation assistance (priority 1).

APPLICATION SCOPE (review-directed): researcher-assisted context annotation
on eligible, in-domain bat bouts — the closest justified use of the frozen
context-classification endpoint. NOT automatic wild-population tagging,
NOT call-end anticipation, NOT stress detection.

PIPELINE (per bout, all recorded together):
    source interval -> validity assessment -> context score ->
    suggested label or ABSTAIN -> reviewer confirmation (human column,
    never auto-filled).

GUARDRAILS (prespecified / declared):
  - the model is the FROZEN audio-only static logistic from the context
    experiment (same fitting procedure; fit-side material only);
  - the score is an UNCALIBRATED model output (probability calibration
    not assessed — declared, not assumed);
  - ABSTENTION BAND: |score - 0.5| < 0.20 -> abstain. This is a DECLARED
    engineering choice for the pilot (not derived from the forecasting
    result and not a validated alarm threshold); its coverage/agreement
    trade-off on the pilot sample is reported;
  - no caller identity required as INPUT is not caller-INVARIANT
    performance: per-caller suggestion agreement is reported;
  - binary C11/C12 scope: the model chooses between these labels only —
    no open-world rejection policy is claimed;
  - output label is "suggested annotated context" — never a statement of
    the animal's intention.

PILOT SAMPLE: 200 eval-side bouts (never in any classifier fit),
deterministic hash order. DESCRIPTIVE AUDIT: agreement between suggestion
and the EXISTING annotation on this sample — an in-domain audit only
(existing labels informed training; it is not an independent validation,
which remains the reviewer-confirmation workflow and the pending
source-reference assessment).

Usage:
    python scripts/context_review_pilot.py [--n 200]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 0
TRAIN_FRAC = 0.7
ABSTAIN_MARGIN = 0.20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    import context_classification as cc
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import filename_block_group

    bouts, _ = cc._load_bouts()
    blocks = sorted({b["block"] for b in bouts})
    train_blocks = {
        bl
        for bl in blocks
        if int(hashlib.sha1(bl.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    for b in bouts:
        b["part"] = "fit" if b["block"] in train_blocks else "eval"
    fit_b = [b for b in bouts if b["part"] == "fit"]
    eval_pool = sorted(
        [b for b in bouts if b["part"] == "eval"],
        key=lambda b: hashlib.sha1(b["file"].encode()).hexdigest(),
    )
    sample = eval_pool[: args.n]

    files = sorted({b["file"] for b in sample})
    contract = ScientificRunContract(
        run_name="context_review_pilot",
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
                produced_by=">=5 valid frames per bout",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(
            files, ["suggestion"] * len(files), [filename_block_group(f) for f in files]
        ),
        eval_split={
            "provenance": "pilot suggestions on eval-side bouts never used in classifier fitting",
            "unit": "recording block",
        },
    )
    enforcement = raise_for_science(contract)

    xtr = np.stack([b["static"] for b in fit_b])
    ytr = np.array([b["context"] for b in fit_b])
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit((xtr - mu) / sd, ytr)
    model_desc = {
        "type": "L2 multinomial logistic (C=1.0, balanced)",
        "features": "static 44-D fingerprint over valid frames",
        "fit_cohort": f"{len(fit_b)} bouts / fit-side blocks only",
        "score": "UNCALIBRATED model output (calibration not assessed)",
    }

    out_dir = ROOT / "outputs" / "context_review_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_suggest = n_agree = 0
    per_caller: dict = {}
    for b in sample:
        x = ((b["static"] - mu) / sd)[None, :]
        p12 = float(clf.predict_proba(x)[0, list(clf.classes_).index(12)])
        abstain = abs(p12 - 0.5) < ABSTAIN_MARGIN
        suggestion = None if abstain else int(clf.predict(x)[0])
        if suggestion is not None:
            n_suggest += 1
            agree = suggestion == b["context"]
            n_agree += int(agree)
            per_caller.setdefault(b["caller"], []).append(int(agree))
        rows.append(
            {
                "file": b["file"],
                "n_valid_frames": int(b["valid"].sum()),
                "score_c12_uncalibrated": round(p12, 4),
                "suggested_context": "" if abstain else suggestion,
                "abstain": "yes" if abstain else "",
                "reviewer_confirmed_context": "",
                "reviewer_notes": "",
            }
        )
    with open(out_dir / "review.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    coverage = n_suggest / max(len(sample), 1)
    agreement = n_agree / max(n_suggest, 1)
    per_caller_summary = {
        str(c): {"n": len(v), "agreement": round(float(np.mean(v)), 3)}
        for c, v in sorted(per_caller.items())
    }
    manifest = {
        "application": "assisted context annotation (suggested annotated "
        "context; reviewer confirmation required; never an intention claim)",
        "model": model_desc,
        "abstention_rule": {
            "band": f"|score - 0.5| < {ABSTAIN_MARGIN}",
            "status": "DECLARED engineering choice for the "
            "pilot; not a validated alarm threshold; "
            "coverage/agreement trade-off reported below",
        },
        "release_enforcement": enforcement,
        "sample": {
            "n": len(sample),
            "selection": "deterministic hash order over eval-side bouts never used in fitting",
        },
        "descriptive_audit": {
            "coverage": round(coverage, 4),
            "suggestion_agreement_with_existing_label": round(agreement, 4),
            "caveat": "in-domain audit only — existing labels informed "
            "training; independent validation = reviewer confirmation "
            "workflow + pending source-reference assessment",
            "per_caller_agreement": per_caller_summary,
        },
        "scope_limits": [
            "binary C11/C12 suggestions; no open-world rejection policy",
            "no caller identity required as input != caller-invariant "
            "performance (per-caller agreement reported)",
            "score is uncalibrated; not a confidence guarantee",
            "validated on curated eligible bouts, not continuous field soundscapes",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["descriptive_audit"], indent=1), flush=True)
    print(
        f"wrote {out_dir}/review.csv ({len(rows)} rows, reviewer columns "
        "empty by design) + manifest.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
