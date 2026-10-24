#!/usr/bin/env python3
"""Complete the previously specified marmoset multi-head predictive evaluation.

ENDPOINTS (fixed hierarchy; reported whatever their sign — completion means
a reported positive/negative/imprecise/unevaluable outcome, not a preferred
sign):

  Four underlying losses on SHARED held-out targets, candidates
  (one negative draw per file, reused across all four computations),
  validity framing and history manipulation:
      L_pop_real      population (UNK) head, original audio
      L_pop_control   population head, history-manipulated audio
      L_caller_real   caller y's head, original audio
      L_caller_control caller y's head, history-manipulated audio
  Derived:
      G_pop = L_pop_control - L_pop_real        (population history gain)
      G_y   = L_caller_control - L_caller_real  (caller history gain)
      D_y   = L_pop_real - L_caller_real        (real-history caller-head benefit)
      T_y   = G_y - G_pop                       (interaction)

  BOUNDARIES (explicit in the artifact):
  - held-out callers have NO trained head: their D_y/T_y stay UNAVAILABLE
    (never zero-filled) — the population-fallback case;
  - own-head benefit is CONDITIONAL PREDICTION evidence (correct caller
    label selects the head), not identification accuracy;
  - scores are CONTRASTIVE CONTINUATION scores (InfoNCE density-ratio
    classification among sampled alternatives), not ordinary acoustic
    perplexity;
  - provenance: caller-supervised checkpoint (contains_caller_supervision:
    true — probe/retrieval-only instrument; never evidence for discovering
    caller structure);
  - evaluation set = the training script's own held-out validation files
    (split reproduced deterministically: holdout_callers seed 0) + the
    held-out callers' files for the population-only case.

Usage:
    python scripts/eval_marmoset_multihead.py [--device xpu:1] [--files-per-caller 4]

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SR = 16000
SEED = 0
BOOT_B = 2000


def _reconstruct_split():
    """Reproduce the training split + head mapping deterministically."""
    import train_species_cpc as ARM_A
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        holdout_callers,
        parse_marmoset_labels_csv,
    )

    data_dir = ARM_A.SPECIES["marmoset"]["default_data"]
    ann = parse_marmoset_labels_csv(data_dir / "labels.csv")
    import dataclasses

    ann = [
        dataclasses.replace(
            a,
            file_name=a.file_name.replace("audio_44k", "audio_16k"),
            recording_group=a.recording_group.replace("audio_44k", "audio_16k"),
        )
        for a in ann
    ]
    split = holdout_callers(ann, n_heldout=2, min_files=4, seed=SEED)
    train_files, val_files = split.train_files, split.val_files
    training = [a for a in ann if (a.recording_group or a.file_name) in train_files]
    train_emitters = sorted({a.emitter for a in training})
    e2i = {e: i for i, e in enumerate(train_emitters)}
    heldout = sorted({a.emitter for a in ann} - set(train_emitters))
    return ann, train_files, val_files, e2i, heldout


def _frames(path: Path, seq_cap: int = 400):
    import numpy as np
    from bioacoustic_pipeline.utils.audio import load_audio

    audio, sr = load_audio(str(path), target_sr=SR)
    frame = int(SR * 0.01)
    n = len(audio) // frame
    if n < 24:
        return None
    x = audio[: min(n, seq_cap) * frame].reshape(min(n, seq_cap), frame)
    return x.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="xpu:1")
    ap.add_argument("--files-per-caller", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import torch

    from bioacoustic_pipeline.diagnostics.scaling_organisation import (
        phase_randomized_surrogate,
    )
    from bioacoustic_pipeline.embeddings.cpc_encoder import EncoderConfig
    from bioacoustic_pipeline.embeddings.multi_head_cpc import (
        UNK_HEAD,
        SharedEncoderMultiHeadCPC,
        per_frame_head_loss,
    )
    from bioacoustic_pipeline.utils.devices import resolve_device

    ck = torch.load(
        ROOT / "checkpoints/marmoset_cpc_multihead/best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    cfg = ck["config"]
    model = SharedEncoderMultiHeadCPC(
        encoder_config=EncoderConfig(
            sample_rate=cfg.sample_rate,
            frame_size_ms=cfg.frame_size_ms,
            hidden_dim=cfg.hidden_dim,
        ),
        steps_ahead=cfg.steps_ahead,
        n_callers=ck["n_callers"],
    )
    model.load_state_dict(ck["model_state_dict"])
    device = resolve_device(args.device)
    model = model.to(device).eval()

    import train_species_cpc as ARM_A

    ann, train_files, val_files, e2i, heldout = _reconstruct_split()
    audio_root = ARM_A.SPECIES["marmoset"]["default_data"] / "data"
    by_caller: dict = {}
    for a in ann:
        key = a.recording_group or a.file_name
        if key in val_files or a.emitter in heldout:
            by_caller.setdefault(a.emitter, set()).add(a.file_name)
    rng = np.random.RandomState(SEED)

    records = []
    for y in sorted(by_caller):
        files = sorted(by_caller[y])[: args.files_per_caller]
        head_key = str(e2i[y]) if y in e2i else None  # None => no trained head
        for f in files:
            p = audio_root / f
            if not p.exists():
                continue
            x = _frames(p)
            if x is None:
                continue
            ctrl = phase_randomized_surrogate(x.astype(np.float64), seed=SEED).astype(np.float32)
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            ct = torch.from_numpy(ctrl).unsqueeze(0).to(device)
            flat = xt.reshape(-1, xt.shape[-1]).shape[0]
            neg_idx = torch.randint(0, flat, (10,), device=device)
            with torch.no_grad():
                l_pop_real = float(
                    per_frame_head_loss(model, xt, head_key=UNK_HEAD, neg_idx=neg_idx).mean()
                )
                l_pop_ctrl = float(
                    per_frame_head_loss(model, ct, head_key=UNK_HEAD, neg_idx=neg_idx).mean()
                )
                if head_key is not None:
                    l_caller_real = float(
                        per_frame_head_loss(model, xt, head_key=head_key, neg_idx=neg_idx).mean()
                    )
                    l_caller_ctrl = float(
                        per_frame_head_loss(model, ct, head_key=head_key, neg_idx=neg_idx).mean()
                    )
                else:
                    l_caller_real = l_caller_ctrl = None
            records.append(
                {
                    "file": f,
                    "caller": int(y),
                    "head": head_key,
                    "heldout_caller": head_key is None,
                    "L_pop_real": l_pop_real,
                    "L_pop_control": l_pop_ctrl,
                    "L_caller_real": l_caller_real,
                    "L_caller_control": l_caller_ctrl,
                }
            )

    def _boot(vals, b=BOOT_B):
        vals = np.asarray(vals, float)
        reps = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(b)]
        return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]

    per_caller = {}
    for y in sorted({r["caller"] for r in records}):
        rs = [r for r in records if r["caller"] == y]
        entry = {
            "n_files": len(rs),
            "G_pop_filelevel": None,
            "G_y": None,
            "G_y_ci": None,
            "D_y": None,
            "T_y": None,
            "L_pop_real": float(np.mean([r["L_pop_real"] for r in rs])),
            "status": None,
        }
        if rs[0]["heldout_caller"]:
            entry["status"] = (
                "held-out caller: no trained head — D_y/T_y UNAVAILABLE (population-fallback case)"
            )
        else:
            gy = [r["L_caller_control"] - r["L_caller_real"] for r in rs]
            entry["G_y"] = float(np.mean(gy))
            entry["G_y_ci"] = _boot(gy)
            entry["D_y"] = float(np.mean([r["L_pop_real"] - r["L_caller_real"] for r in rs]))
        per_caller[str(y)] = entry
    g_pop_all = [r["L_pop_control"] - r["L_pop_real"] for r in records]
    g_pop_mean = float(np.mean(g_pop_all))
    for y, e in per_caller.items():
        if e.get("G_y") is not None:
            e["T_y"] = round(e["G_y"] - g_pop_mean, 4)

    report = {
        "task": "marmoset multi-head predictive evaluation (previously "
        "specified endpoints, completed once; no selection on results)",
        "checkpoint": {
            "path": "checkpoints/marmoset_cpc_multihead/best_model.pt",
            "epoch": ck.get("epoch"),
            "loss_version": ck.get("loss_version"),
            "n_callers": ck["n_callers"],
            "steps_ahead": cfg.steps_ahead,
            "hidden_dim": cfg.hidden_dim,
            "provenance": "caller-supervised (contains_caller_supervision: "
            "true — probe/retrieval-only instrument)",
        },
        "endpoint_definitions": {
            "G_pop": "L_pop_control - L_pop_real (population history gain)",
            "G_y": "L_caller_control - L_caller_real (caller history gain)",
            "D_y": "L_pop_real - L_caller_real (real-history caller-head "
            "benefit — CONDITIONAL prediction, correct label selects head)",
            "T_y": "G_y - G_pop (interaction)",
            "control": "phase-randomised history manipulation (spectrum "
            "preserved), identical negative draw across all four losses "
            "per file",
        },
        "boundaries": [
            "held-out callers: D_y/T_y UNAVAILABLE, never zero-filled",
            "own-head benefit is conditional prediction evidence, NOT identification accuracy",
            "scores are contrastive continuation scores (InfoNCE density-"
            "ratio classification among sampled alternatives), NOT ordinary "
            "acoustic perplexity",
        ],
        "n_eval_files": len(records),
        "G_pop": round(g_pop_mean, 4),
        "G_pop_ci": [round(v, 4) for v in _boot(g_pop_all)],
        "per_caller": per_caller,
        "records": records,
    }
    out = ROOT / "outputs" / "marmoset_multihead_eval.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                "n_files": len(records),
                "G_pop": report["G_pop"],
                "G_pop_ci": report["G_pop_ci"],
                "per_caller": {
                    y: {"G_y": e["G_y"], "D_y": e["D_y"], "T_y": e["T_y"], "status": e["status"]}
                    for y, e in per_caller.items()
                },
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
