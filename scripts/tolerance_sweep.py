#!/usr/bin/env python3
"""Onset/offset F1 tolerance sweep — MeerKAT vs InfantMarmosetsVox.

Closes the tolerance-confound in the router claim: the headline F1s used different
tolerances (MeerKAT 100ms, marmoset 200ms). This reports onset+offset F1 at 50/100/200ms
for BOTH so the comparison is apples-to-apples.

  MeerKAT: loads the cached eval fixture (checkpoints/hecb/test_fixture.pkl) → instant.
  InfantMarmosetsVox: trains the CRNN once, sweeps over test files.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from bioacoustic_pipeline.segmentation.nbd_evaluation import prf_one_to_one  # noqa: E402

TOLS = [50.0, 100.0, 200.0]


def _prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-8), p, r


def sweep_meerkat():
    """Load cached fixture → onset/offset F1 at each tolerance."""
    with open(ROOT / "checkpoints" / "hecb" / "test_fixture.pkl", "rb") as f:
        fixture = pickle.load(f)
    print(f"=== MeerKAT (cached fixture, {len(fixture)} clips) ===")
    print(f"{'tol(ms)':>8} {'onset F1':>9} {'onset P':>8} {'onset R':>8} | "
          f"{'offset F1':>9} {'offset P':>8} {'offset R':>8}")
    for tol in TOLS:
        on = [0, 0, 0]
        off = [0, 0, 0]
        for fx in fixture:
            crnn_on = [e[0] for e in fx["crnn_events"]]
            crnn_off = [e[1] for e in fx["crnn_events"]]
            r = prf_one_to_one(crnn_on, fx["gt_on"], tol=tol)
            on[0] += r.tp; on[1] += r.fp; on[2] += r.fn
            r = prf_one_to_one(crnn_off, fx["gt_off"], tol=tol)
            off[0] += r.tp; off[1] += r.fp; off[2] += r.fn
        of, op, orr = _prf(*on)
        ff, fp, fr = _prf(*off)
        print(f"{int(tol):>8} {of:>9.3f} {op:>8.3f} {orr:>8.3f} | {ff:>9.3f} {fp:>8.3f} {fr:>8.3f}")


def sweep_infant_marmoset(n_train, n_test, epochs):
    import torch  # noqa: PLC0415
    import train_hecb_infant_marmoset as IM  # reuse CRNN train + inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = IM.sample_files(n_train + n_test, 0)
    train_files, test_files = files[:n_train], files[n_train:]
    print(f"\n=== InfantMarmosetsVox (training CRNN: {len(train_files)} files, "
          f"{epochs} epochs) ===", flush=True)
    crnn = IM.train_crnn(train_files, device, epochs)
    print(f"  sweeping onset/offset F1 over {len(test_files)} test files...", flush=True)
    acc = {t: {"on": [0, 0, 0], "off": [0, 0, 0]} for t in TOLS}
    for fn, wav, labels in test_files:
        try:
            audio, _ = IM.load_audio(str(wav), target_sr=IM.SR)
        except Exception:  # noqa: BLE001
            continue
        gt = IM.get_call_spans(labels)
        gt_on = [s * 1000 for s, _e, _ct, _c in gt]
        gt_off = [e * 1000 for _s, e, _ct, _c in gt]
        proba = IM._crnn_proba(crnn, audio, device)
        events = IM.proba_to_events(proba)
        ev_on = [e[0] for e in events]
        ev_off = [e[1] for e in events]
        for tol in TOLS:
            r = prf_one_to_one(ev_on, gt_on, tol=tol)
            acc[tol]["on"][0] += r.tp; acc[tol]["on"][1] += r.fp; acc[tol]["on"][2] += r.fn
            r = prf_one_to_one(ev_off, gt_off, tol=tol)
            acc[tol]["off"][0] += r.tp; acc[tol]["off"][1] += r.fp; acc[tol]["off"][2] += r.fn
    print(f"\n{'tol(ms)':>8} {'onset F1':>9} {'onset P':>8} {'onset R':>8} | "
          f"{'offset F1':>9} {'offset P':>8} {'offset R':>8}")
    for tol in TOLS:
        of, op, orr = _prf(*acc[tol]["on"])
        ff, fp, fr = _prf(*acc[tol]["off"])
        print(f"{int(tol):>8} {of:>9.3f} {op:>8.3f} {orr:>8.3f} | {ff:>9.3f} {fp:>8.3f} {fr:>8.3f}")


def main():
    ap = argparse.ArgumentParser(description="Onset/offset F1 tolerance sweep")
    ap.add_argument("--skip-marmoset", action="store_true")
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--crnn-epochs", type=int, default=25)
    args = ap.parse_args()
    sweep_meerkat()
    if not args.skip_marmoset:
        sweep_infant_marmoset(args.n_train, args.n_test, args.crnn_epochs)
    print("\nTOLERANCE SWEEP DONE")


if __name__ == "__main__":
    main()
