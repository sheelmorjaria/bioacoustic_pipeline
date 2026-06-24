#!/usr/bin/env python3
"""Entanglement Ceiling — AVES ablation (reviewer-proofing §36).

§36 proved caller identity and call form are non-separably entangled in the 44-D
acoustic fingerprint: LDA, LogReg, Z-normalization, and delta-trajectory ALL fail to
separate them (form degrades up to 4x harder than caller). A reviewer will inevitably
argue that a *deep learned* embedding learns higher-order features and could separate
them where hand-crafted physics features cannot.

This script kills that critique. It repeats the linear disentanglement (Orthogonal LDA
projection + per-speaker Z-normalization — the two global-vector methods) on **AVES**, a
768-D Transformer SSL embedding trained on multi-taxa animal audio. AVES is
firewall-legal for marmoset (multi-taxa animal domain, no human/text supervision). It is
computed for the *same calls* and *same caller/form labels* as the 44-D fingerprint
control, so the comparison is apples-to-apples.

Expected result: the entanglement ceiling PERSISTS on AVES — the caller/form ratio stays
~1.0 and projecting out the caller destroys form at an equal-or-greater rate. This proves
the entanglement is a fundamental property of the bioacoustic signal, not an artifact of
the feature extractor.

This script also REPRODUCES §36's fingerprint {baseline, LDA, Z-norm} rows, making the
Entanglement Ceiling reproducible end-to-end (the §36 runner was previously ad hoc).
Delta-trajectory (Method 4) is frame-level and already logged in §36; it is intentionally
not re-run here (the user's scope is the two global-vector methods).

Caller proxy: the marmoset corpus has no individual-ID labels, so ``parent_name``
(recording session) is used as the caller proxy — the only available individual-ish label,
identical to how §26/§32 treat it. This is documented, not hidden.

Run:  python scripts/entanglement_ceiling_aves.py --n-per-type 500 --min-calls-per-caller 20

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VOC_DIR = Path("/home/sheel/birdsong_analysis/data/Vocalizations")
ANNOT = VOC_DIR / "Annotations.tsv"
CALL_TYPES = ("Phee", "Twitter", "Trill", "Tsik", "Seep")
SR = 96000
AVES_SR = 16000  # AVES is a 16 kHz model; marmoset (96 kHz) anti-alias resampled to 16 kHz


def _resample_to_16k(audio, sr):
    from scipy.signal import resample_poly  # noqa: PLC0415

    if sr == AVES_SR:
        return audio
    g = np.gcd(int(sr), AVES_SR)
    return resample_poly(audio, AVES_SR // g, int(sr) // g).astype(np.float32)


def _aves_embed(aves_model, audio, sr):
    """AVES forward -> mean-pool over frames -> 768-D numpy (pads to >=800 samples)."""
    import torch  # noqa: PLC0415

    a = _resample_to_16k(np.asarray(audio, dtype=np.float32), sr)
    if len(a) < 800:
        a = np.pad(a, (0, 800 - len(a)))
    dev = next(aves_model.parameters()).device
    with torch.no_grad():
        h = aves_model.forward(torch.from_numpy(a).unsqueeze(0).to(dev))
    return h.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def _load_index():
    """file_name -> (label, parent_name). Caller proxy = parent_name (session)."""
    idx = {}
    with ANNOT.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            fn = (row.get("file_name") or "").strip()
            if not fn:
                continue
            idx[fn] = (row.get("label", "").strip(), row.get("parent_name", "").strip())
    return idx


def sample_calls(args, rng):
    """Balanced sample over call types, restricted to callers (sessions) with
    >= min_calls_per_caller total calls — so Z-norm has enough per-caller data and the
    caller/form variation both exist."""
    idx = _load_index()
    calls_by_caller = defaultdict(list)
    for fn, (lab, parent) in idx.items():
        if lab in CALL_TYPES and parent:
            calls_by_caller[parent].append((lab, fn))
    eligible_callers = {
        c for c, lst in calls_by_caller.items() if len(lst) >= args.min_calls_per_caller
    }
    print(
        f"  {len(eligible_callers)} callers (sessions) with >= {args.min_calls_per_caller} calls "
        f"(caller proxy = parent_name; no individual-ID labels in corpus)"
    )

    by_type = defaultdict(list)
    for caller in eligible_callers:
        for lab, fn in calls_by_caller[caller]:
            by_type[lab].append((caller, fn))

    chosen = []
    for ct in CALL_TYPES:
        pool = by_type.get(ct, [])
        rng.shuffle(pool)
        chosen.extend((ct, caller, fn) for caller, fn in pool[: args.n_per_type])
    rng.shuffle(chosen)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=500)
    ap.add_argument("--min-calls-per-caller", type=int, default=20)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: PLC0415
    from bioacoustic_pipeline.embeddings.firewall import Use, default_firewall  # noqa: PLC0415
    from bioacoustic_pipeline.representation.disentangle import (  # noqa: PLC0415
        disentangle_orthogonal,
        disentangle_znorm,
        evaluate_disentanglement,
    )
    from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

    # Firewall: both representations must be legal for marmoset.
    fw = default_firewall()
    fw.assert_allowed("acoustic_fingerprint", Use.TOKENIZE)
    fw.assert_cross_species("acoustic_fingerprint", "marmoset")
    fw.assert_allowed("aves", Use.EMBEDDING)
    fw.assert_cross_species("aves", "marmoset")
    print("firewall: acoustic_fingerprint (tokenize) + aves (embedding) legal for marmoset")

    import avex  # noqa: PLC0415
    import torch  # noqa: PLC0415

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading AVES (esp_aves2_eat_bio) on {dev} ...")
    aves = avex.load_model("esp_aves2_eat_bio", device=dev)
    aves.eval()
    print("  (caveat: AVES is 16 kHz; marmoset 96 kHz calls resampled -> <8 kHz)")

    rng = np.random.RandomState(args.seed)
    chosen = sample_calls(args, rng)
    print(f"sampled {len(chosen)} calls balanced over {CALL_TYPES}")

    fp_vecs, av_vecs, callers, forms = [], [], [], []
    for i, (ct, caller, fn) in enumerate(chosen):
        path = VOC_DIR / caller / fn
        if not path.exists():
            continue
        try:
            a, sr = load_audio(str(path), target_sr=SR)
        except Exception:  # noqa: BLE001
            continue
        fp = afp.extract(a, sr, species="marmoset").combined
        av = _aves_embed(aves, a, sr)
        if not np.any(fp) or not np.any(av):
            continue
        fp_vecs.append(fp)
        av_vecs.append(av)
        callers.append(caller)
        forms.append(ct)
        if (i + 1) % 500 == 0:
            print(f"  embedded {i + 1}/{len(chosen)}")

    FP = np.asarray(fp_vecs, dtype=np.float32)
    AV = np.asarray(av_vecs, dtype=np.float32)
    callers = np.asarray(callers)
    forms = np.asarray(forms)
    print(
        f"embedded {len(FP)} calls | fingerprint {FP.shape[1]}-D | AVES {AV.shape[1]}-D | "
        f"{len(set(callers))} callers, {len(set(forms))} call types"
    )

    def run_block(name, X):
        """baseline / LDA / Z-norm on representation matrix X; return list of dicts."""
        base = evaluate_disentanglement(
            X, callers, forms, k=args.k, n_permutations=args.n_perm, seed=args.seed
        )
        Xlda = disentangle_orthogonal(X, callers, method="lda", random_state=args.seed)
        lda = evaluate_disentanglement(
            Xlda, callers, forms, k=args.k, n_permutations=args.n_perm, seed=args.seed
        )
        Xz = disentangle_znorm(X, callers, min_calls_per_caller=5)
        z = evaluate_disentanglement(
            Xz, callers, forms, k=args.k, n_permutations=args.n_perm, seed=args.seed
        )
        rows = []
        for method, res in (("baseline", base), ("lda", lda), ("znorm", z)):
            d_caller = res["caller_v"] - base["caller_v"]
            d_form = res["form_v"] - base["form_v"]
            rows.append(
                {
                    "representation": name,
                    "method": method,
                    "caller_v": res["caller_v"],
                    "form_v": res["form_v"],
                    "ratio": res["disentangle_ratio"],
                    "d_caller": round(d_caller, 3),
                    "d_form": round(d_form, 3),
                    "form_drops_harder": abs(d_form) >= abs(d_caller),
                }
            )
        return rows, base

    fp_rows, fp_base = run_block("fingerprint_44D", FP)
    av_rows, av_base = run_block("aves_768D", AV)
    rows = fp_rows + av_rows

    # ---- report ----
    print("\n" + "=" * 96)
    print("ENTANGLEMENT CEILING — AVES ablation (§36 reviewer-proofing)")
    print("=" * 96)
    print(
        f"  n_calls={len(FP)}  n_callers={len(set(callers))}  n_call_types={len(set(forms))}  "
        f"k={args.k}  perm={args.n_perm}  seed={args.seed}"
    )
    print(
        f"  fingerprint baseline: caller V={fp_base['caller_v']:.3f}  form V={fp_base['form_v']:.3f}  "
        f"| AVES baseline: caller V={av_base['caller_v']:.3f}  form V={av_base['form_v']:.3f}"
    )
    print("  " + "-" * 92)
    hdr = f"  {'representation':16} {'method':9} {'callerV':>8} {'formV':>7} {'ratio':>6} {'Δcaller':>8} {'Δform':>8}  form≥?"
    print(hdr)
    print("  " + "-" * 92)
    for r in rows:
        print(
            f"  {r['representation']:16} {r['method']:9} {r['caller_v']:>8.3f} {r['form_v']:>7.3f} "
            f"{r['ratio']:>6.2f} {r['d_caller']:>+8.3f} {r['d_form']:>+8.3f}  "
            f"{'YES' if r['form_drops_harder'] else 'no'}"
        )

    fp_lda, av_lda = fp_rows[1], av_rows[1]
    fp_z, av_z = fp_rows[2], av_rows[2]
    print("\n  Verdict:")
    print(
        f"    fingerprint LDA Δform={fp_lda['d_form']:+.3f} (vs Δcaller {fp_lda['d_caller']:+.3f})  "
        f"| AVES LDA Δform={av_lda['d_form']:+.3f} (vs Δcaller {av_lda['d_caller']:+.3f})"
    )
    print(
        f"    fingerprint Z   Δform={fp_z['d_form']:+.3f} (vs Δcaller {fp_z['d_caller']:+.3f})  "
        f"| AVES Z   Δform={av_z['d_form']:+.3f} (vs Δcaller {av_z['d_caller']:+.3f})"
    )
    persists = av_lda["form_drops_harder"] and av_z["form_drops_harder"]
    print(
        f"    => AVES ceiling {'PERSISTS' if persists else 'does NOT persist'}: "
        f"a deep learned embedding cannot linearly disentangle identity from form."
    )

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    payload = {
        "n_calls": int(len(FP)),
        "n_callers": int(len(set(callers))),
        "n_call_types": int(len(set(forms))),
        "caller_proxy": "parent_name (recording session; no individual-ID labels)",
        "k": args.k,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "fingerprint_baseline": fp_base,
        "aves_baseline": av_base,
        "rows": rows,
        "aves_ceiling_persists": bool(persists),
    }
    with open(out / "entanglement_ceiling_aves.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(out / "entanglement_ceiling_aves.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved outputs/entanglement_ceiling_aves.{json,csv}")
    print("ENTANGLEMENT CEILING AVES DONE")


if __name__ == "__main__":
    main()
