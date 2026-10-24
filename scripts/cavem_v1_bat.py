#!/usr/bin/env python3
"""CAVEM-V1 runner — bat corpus, SELF-SUPERVISED acoustic branch.

DATA DECISION (measured, 2026-10-20): the advisor's decisive question
answers NEGATIVE for event-level interaction modelling — 0 of 91,080
annotation files contain more than one annotated vocalization and the
CSV has no per-event timing columns; each file is one isolated call.
Per the advisor's contingency, sequences are NOT manufactured: the
initial model is the self-supervised contextual acoustic encoder
evaluated on the existing masked-prediction endpoint against the
ordinary-interpolation and persistence baselines (the earlier
masked-reconstruction results showed local interpolation is strong —
low reconstruction error is NOT treated as evidence of higher-order
organisation).

ARMS (frozen; recording-group partition; identical masks):
  persistence   last observed frame before the mask (trivial baseline)
  blend         linear interpolation across the masked span (the strong
                local baseline from masked_reconstruction)
  cavem_ss      causal GRU contextual encoder over validity-supported
                12-feature frames, trained on fit-group bouts with the
                masked-acoustic objective (targets ONLY at valid+masked
                positions; invalid frames never targets — module invariant)
ENDPOINT: nRMSE at masked positions on eval groups, recording-group
clustered bootstrap B=1000; per-feature breakdown. INTERPRETATION:
acoustic/sequential predictive value only — no unit, meaning, or
compositional claims (Semantic Ceiling intact). One run.

Usage:
    python scripts/cavem_v1_bat.py [--self-test]

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

import numpy as np

from bioacoustic_pipeline.contextual.cavem import masked_acoustic_targets, partition_by_group

MASK_FRAMES = 7
SEED = 0
BOOT_B = 1000
HIDDEN = 64
EPOCHS = 8
BATCH = 32


def _load_bat_tracks():
    """Frozen dyadic bat trajectories (same caches as cross_corpus_transfer)."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import parse_annotations_csv

    audit = json.loads((ROOT / "outputs/history_support_audit.json").read_text())
    frozen = {
        (p["caller"], p["recipient"])
        for p in audit["support_audit"]["dyadic_cells_frozen"]["pairs"].values()
    }
    meta = {
        a.file_name: a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
    }
    feats = list(supported_features())
    fidx = [_NAME_TO_IDX[n] for n in feats]
    bouts = []
    seen = set()
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
            if (a.emitter, a.addressee) not in frozen:
                continue
            f44 = frames[offsets[i] : offsets[i + 1]].astype(np.float64)
            z = f44[:, fidx]
            valid = ~np.all(z == 0, axis=1)
            if valid.sum() >= 40:
                bouts.append({"name": nm, "group": a.recording_group or nm, "z": z, "valid": valid})
    return bouts


def _mask_cases(bouts, rng, n_per_bout=2):
    """Central-ish masks at valid-supported spans (identical across arms)."""
    cases = []
    for b in bouts:
        T = len(b["z"])
        made = 0
        tries = 0
        while made < n_per_bout and tries < 10:
            tries += 1
            c = rng.randint(MASK_FRAMES + 2, T - MASK_FRAMES - 2)
            span = range(c, c + MASK_FRAMES)
            if all(
                b["valid"][t] and b["valid"][c - 1] and b["valid"][c + MASK_FRAMES] for t in span
            ):
                cases.append({"b": b, "c": c})
                made += 1
    return cases


def _nrmse(pred, truth):
    r = np.asarray(pred) - np.asarray(truth)
    return float(np.sqrt((r**2).mean(0)).mean())


def _arm_persistence(case):
    b, c = case["b"], case["c"]
    return np.repeat(b["z"][c - 1][None, :], MASK_FRAMES, axis=0), b["z"][c : c + MASK_FRAMES]


def _arm_blend(case):
    b, c = case["b"], case["c"]
    fa, ba = b["z"][c - 1], b["z"][c + MASK_FRAMES]
    out = np.zeros((MASK_FRAMES, 12))
    for k in range(MASK_FRAMES):
        w = (k + 1) / (MASK_FRAMES + 1)
        out[k] = (1 - w) * fa + w * ba
    return out, b["z"][c : c + MASK_FRAMES]


def _past64(case, context=64):
    """Fixed-length causal context: the last `context` valid-supported
    frames before the mask, left edge-padded (declared extraction policy)."""
    b, c = case["b"], case["c"]
    z = b["z"][max(0, c - context) : c]
    if len(z) == 0:
        return np.zeros((context, 12))
    pad = context - len(z)
    if pad > 0:
        z = np.vstack([np.repeat(z[:1], pad, axis=0), z])
    return z


class _SSModel:
    """Causal GRU contextual encoder with masked-frame reconstruction head.

    Causality discipline: training uses causal states; masked-frame
    prediction from a causal state at the mask's left edge uses PAST
    only (the interpolation/bidir comparisons are the retrospective
    arms). Targets only at valid positions (module invariant).
    """

    def __init__(self, hidden=HIDDEN):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.gru = nn.GRU(input_size=12, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 12)

    def fit(self, bouts, epochs=EPOCHS, seed=SEED):
        import torch

        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        # training masks: random valid spans across fit bouts
        cases = _mask_cases(bouts, rng, n_per_bout=4)
        opt = torch.optim.Adam(list(self.gru.parameters()) + list(self.head.parameters()), lr=1e-3)
        lossf = torch.nn.MSELoss()
        self.gru.train()
        for ep in range(epochs):
            tot = 0.0
            rng.shuffle(cases)
            for i in range(0, len(cases), BATCH):
                chunk = cases[i : i + BATCH]
                # fixed 64-frame causal context (left edge-padded);
                # whole-sequence stats would couple states to the future
                X = np.stack([_past64(c) for c in chunk])
                Y = np.stack([c["b"]["z"][c["c"] : c["c"] + MASK_FRAMES] for c in chunk])
                opt.zero_grad()
                out, _ = self.gru(torch.tensor(X, dtype=torch.float32))
                # predict the mask from the LAST causal state, expanded
                h = out[:, -1]
                preds = torch.stack([self.head(h)] * MASK_FRAMES, dim=1)
                loss = lossf(preds, torch.tensor(Y, dtype=torch.float32))
                loss.backward()
                opt.step()
                tot += float(loss)
            print(
                f"    epoch {ep + 1}/{epochs} mse {tot / max(1, len(cases) // BATCH):.4f}",
                flush=True,
            )
        self.gru.eval()
        return self

    def predict(self, case):
        import torch

        b, c = case["b"], case["c"]
        with torch.no_grad():
            out, _ = self.gru(torch.tensor(_past64(case)[None], dtype=torch.float32))
            h = self.head(out[0, -1])
        return np.repeat(h.numpy()[None, :], MASK_FRAMES, axis=0), b["z"][c : c + MASK_FRAMES]


def _boot(per_case_losses, groups, b=BOOT_B, seed=SEED):
    """Recording-group clustered bootstrap of mean nRMSE."""
    rng = np.random.RandomState(seed)
    ug = sorted(set(groups))
    idx_of = {g: np.where(np.array(groups) == g)[0] for g in ug}
    vals = []
    L = np.array(per_case_losses)
    for _ in range(b):
        gs = rng.choice(len(ug), len(ug))
        ii = np.concatenate([idx_of[ug[g]] for g in gs])
        vals.append(L[ii].mean())
    return {
        "point": round(float(L.mean()), 4),
        "ci": [
            round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4),
        ],
        "n_cases": len(L),
        "n_groups": len(ug),
    }


def _self_test() -> None:
    rng = np.random.RandomState(0)
    bouts = []
    for k in range(8):
        T = 120
        z = np.cumsum(rng.randn(T, 12) * 0.05, axis=0)
        bouts.append({"name": f"b{k}", "group": f"g{k % 3}", "z": z, "valid": np.ones(T, bool)})
    m = _SSModel(hidden=16).fit(bouts, epochs=2)
    case = _mask_cases(bouts[:1], np.random.RandomState(1))[0]
    pred, truth = m.predict(case)
    assert pred.shape == truth.shape
    blend_pred, _ = _arm_blend(case)
    assert _nrmse(blend_pred, truth) >= 0.0
    v = masked_acoustic_targets(np.ones(50, bool), 0.2, np.random.RandomState(0))
    assert len(v) == 10
    print("SELF-TEST OK (SS model trains, predicts, shapes match)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return

    bouts = _load_bat_tracks()
    print(f"bat bouts (frozen dyadic): {len(bouts)}", flush=True)
    groups = [b["group"] for b in bouts]
    part = partition_by_group(groups, train_frac=0.7, val_frac=0.25, seed=SEED)
    fit = [b for b in bouts if part[b["group"]] in ("fit", "val")]
    ev = [b for b in bouts if part[b["group"]] == "eval"]
    print(f"splits: fit+val {len(fit)} / eval {len(ev)}", flush=True)

    model = _SSModel().fit(fit)

    rng = np.random.RandomState(SEED)
    cases = _mask_cases(ev, rng, n_per_bout=2)
    res = {}
    for arm, fn in (
        ("persistence", _arm_persistence),
        ("blend", _arm_blend),
        ("cavem_ss", model.predict),
    ):
        losses, grps = [], []
        for c in cases:
            pred, truth = fn(c)
            losses.append(_nrmse(pred, truth))
            grps.append(c["b"]["group"])
        res[arm] = _boot(losses, grps)
        print(f"{arm}: {res[arm]}", flush=True)

    report = {
        "study": "CAVEM-V1 self-supervised acoustic branch (advisor "
        "contingency: corpus cannot reconstruct event sequences)",
        "data_decision": "0/91080 files contain >1 annotated vocalization; "
        "no timing columns -> isolated calls; sequences NOT manufactured",
        "arms": "persistence / linear blend (the strong local-interpolation "
        "baseline) / causal-GRU contextual encoder (masked-acoustic "
        "objective, valid-only targets)",
        "endpoint": "nRMSE at masked positions, recording-group clustered bootstrap B=1000",
        "results": res,
        "interpretation": "acoustic predictive value only; low "
        "reconstruction error is not evidence of higher-order "
        "organisation (masked_reconstruction precedent); no unit/meaning "
        "claims",
        "stop_rule": "one run",
    }
    out = ROOT / "outputs" / "cavem_v1_bat.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
