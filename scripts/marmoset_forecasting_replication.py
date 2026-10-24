#!/usr/bin/env python3
"""Cross-corpus PROCEDURE replication of recent-history acoustic forecasting.

HYPOTHESIS (prespecified): recent acoustic history improves prospective
prediction of supported acoustic features in marmoset vocalisations, as it
did in the bat analysis. Primary contrast (original direction preserved):

    Delta_history = nRMSE(state) - nRMSE(state+derivative)   [positive = improvement]

History-matched control retained: state+derivative vs state+previous-state
(informationally equivalent parameterisations; a derivative win is a
parameterisation result, not new information).

PROCEDURE REPLICATION, NOT MODEL TRANSFER: scaler statistics and Ridge
coefficients are REFIT on marmoset training material under the original
declared fitting policy (fit-scaler from FIT files; per-arm alpha on VAL;
final refit FIT+VAL; EVAL untouched). Bat-fitted coefficients are never
applied and called a replication.

MATERIAL: InfantMarmosetsVox. Trajectories: 40 ms / 10 ms fingerprint
framing (identical physical scale to the bat study); the supported
12-feature subset (schema v2); validity = the fingerprint's own gate.
Recording identity = the date block of the source file name (the corpus's
recording-session proxy — extracted calls trace to longer recordings);
every call of one recording stays in one partition (hash split 70/30;
within the train side, the original 75/25 FIT/VAL file split).
Prospective horizon: h=4 frames (h*hop >= win — same no-overlap rule);
eligibility identical (valid[t-1] & valid[t] & valid[t+h]).

CLAIM SCOPE: this tests generality of the PREDICTIVE result, not the
intra-call word hypothesis. The marmoset material has already been used by
this programme (crosstaxon, multihead) though never for this endpoint;
the result is a new prespecified analysis of an established corpus, not
fresh-data confirmation.

Usage:
    python scripts/marmoset_forecasting_replication.py [--jobs 24]

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

SR = 16000
WIN_MS, HOP_MS = 40.0, 10.0
MIN_SECONDS = 1.0
MAX_CALLS_PER_CALLER = 80
PRIMARY_H = 4
TRAIN_FRAC = 0.7
SEED = 0


def _recording_group(file_name: str) -> str:
    """Date block of the source file = recording-session proxy."""
    stem = Path(file_name).name
    digits = "".join(ch for ch in stem[:12] if ch.isdigit())
    return f"day_{digits[:8]}" if len(digits[:8]) == 8 else stem


def _extract(args):
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.utils.audio import load_audio

    path, name, caller = args
    audio, sr = load_audio(str(path), target_sr=SR)
    win, hop = int(sr * WIN_MS / 1000), int(sr * HOP_MS / 1000)
    rows, i = [], 0
    while i * hop + win <= len(audio) and len(rows) < 400:
        v = np.asarray(
            afp.extract(audio[i * hop : i * hop + win], sr, species="marmoset").combined,
            dtype=np.float64,
        )
        rows.append(v)
        i += 1
    if len(rows) < int(MIN_SECONDS * SR / (SR * HOP_MS / 1000)):
        return None
    return name, caller, np.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()

    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    import train_species_cpc as ARM_A
    from acoustic_dynamics_report import ALPHAS as ALPHA_GRID, ARMS, run_forecast
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
        _NAME_TO_IDX,
        first_derivative,
        supported_features,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import parse_marmoset_labels_csv
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )

    data_dir = ARM_A.SPECIES["marmoset"]["default_data"]
    audio_root = data_dir / "data"
    ann = parse_marmoset_labels_csv(data_dir / "labels.csv")
    import dataclasses

    ann = [
        dataclasses.replace(
            a,
            file_name=a.file_name.replace("audio_44k", "audio_16k"),
        )
        for a in ann
    ]
    # deterministic per-caller cap, duration-eligible material only
    rng = np.random.RandomState(SEED)
    by_caller: dict = {}
    for a in ann:
        p = audio_root / a.file_name
        if p.exists():
            by_caller.setdefault(a.emitter, []).append(a)
    chosen = []
    for y in sorted(by_caller):
        files = sorted(by_caller[y], key=lambda a: a.file_name)
        rng.shuffle(files)
        seen: set = set()
        for a in files:
            if a.file_name in seen:
                continue
            seen.add(a.file_name)
            chosen.append((audio_root / a.file_name, a.file_name, y))
            if len(seen) >= MAX_CALLS_PER_CALLER:
                break
    print(f"extraction targets: {len(chosen)} calls", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = [r for r in ex.map(_extract, chosen, chunksize=4) if r]

    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]
    bouts = []
    for name, caller, frames in results:
        valid = ~np.all(frames == 0, axis=1)
        sub = frames[:, fidx]
        derivs = np.full_like(sub, np.nan)
        for j in range(len(feats)):
            derivs[:, j] = first_derivative(sub[:, j], valid, dt=HOP_MS / 1000.0)
        bouts.append(
            {
                "file": name,
                "caller": caller,
                "group": _recording_group(name),
                "part": None,
                "s": sub,
                "d": derivs,
                "valid": valid,
            }
        )

    groups = sorted({b["group"] for b in bouts})
    train_groups = {
        g
        for g in groups
        if int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    for b in bouts:
        b["part"] = "fit-side" if b["group"] in train_groups else "eval"
    files = sorted({b["file"] for b in bouts})
    sorted({b["group"] for b in bouts})
    contract = ScientificRunContract(
        run_name="marmoset_forecasting_replication",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        masks={
            "source_validity": MaskProvenance(
                kind="source_validity",
                produced_by="fingerprint absolute gate",
                win_ms=WIN_MS,
                hop_ms=HOP_MS,
                amplitude_stage="native waveform",
            ),
            "detected_activity": MaskProvenance(
                kind="detected_activity",
                produced_by="valid-span frames",
                win_ms=WIN_MS,
                hop_ms=HOP_MS,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(
            files,
            [next(b["part"] for b in bouts if b["file"] == f) for f in files],
            [_recording_group(f) for f in files],
        ),
        eval_split={
            "provenance": "recording-group (date-block) hash split "
            f"{int(TRAIN_FRAC * 100)}/{int((1 - TRAIN_FRAC) * 100)}",
            "unit": "recording group",
        },
    )
    enforcement = raise_for_science(contract)
    print(
        f"bouts: {len(bouts)} | groups {len(groups)} "
        f"(train-side {sum(1 for b in bouts if b['part'] == 'fit-side')} / "
        f"eval {sum(1 for b in bouts if b['part'] == 'eval')})",
        flush=True,
    )

    import acoustic_dynamics_report as adr

    rows_all = adr.assemble_horizon(bouts, PRIMARY_H)[0]
    # rows are per ELIGIBLE POSITION; the mask maps through the row->bout
    # index exactly as the original study did
    eval_mask = np.array([bouts[f]["part"] == "eval" for f in rows_all["files"]])
    train_rows = {k: v[~eval_mask] for k, v in rows_all.items() if k != "same_span"}
    eval_rows = {k: v[eval_mask] for k, v in rows_all.items() if k != "same_span"}
    out = run_forecast(train_rows, eval_rows, alphas=ALPHA_GRID, n_boot=2000)
    delta_history = out["arms"]["state"]["aggregate"] - out["arms"]["state+deriv"]["aggregate"]
    d_ctrl = out["bootstrap_deltas"]["state+deriv minus state+prev"]

    report = {
        "claim_scope": "generality of the PREDICTIVE result (recent-history "
        "forecasting); NOT the intra-call word hypothesis",
        "replication_type": "procedure replication (scaler + Ridge refit on "
        "marmoset training material; model transfer never performed)",
        "primary_contrast": "Delta_history = nRMSE(state) - "
        "nRMSE(state+derivative); POSITIVE = improvement (original "
        "direction preserved)",
        "hypothesis": "recent acoustic history improves prospective "
        "prediction of supported acoustic features in marmoset "
        "vocalisations, as in the bat analysis",
        "release_enforcement": enforcement,
        "counts": {
            "bouts": len(bouts),
            "groups": len(groups),
            "eval_rows": out["n_eval_rows"],
            "callers": len({b["caller"] for b in bouts}),
            "features": list(feats),
        },
        "arms": {a: out["arms"][a]["aggregate"] for a in ARMS},
        "primary_delta_history": round(float(delta_history), 4),
        "primary_bootstrap": out["bootstrap_deltas"]["state+deriv minus state"],
        "history_matched_control": d_ctrl,
        "status_note": "new prespecified analysis of an established corpus "
        "(marmoset material used by earlier programme arms, never for this "
        "endpoint); not fresh-data confirmation",
    }
    out_path = ROOT / "outputs" / "marmoset_forecasting_replication.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "arms",
                    "primary_delta_history",
                    "primary_bootstrap",
                    "history_matched_control",
                    "counts",
                )
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
