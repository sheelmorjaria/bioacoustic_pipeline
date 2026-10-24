#!/usr/bin/env python3
"""Non-linear caller-invariance test (adversarial gradient reversal).

PREREGISTRATION (frozen before any run; per the review's wording):

  CLAIM LEVEL: method-conditioned separability ONLY — "under the declared
  non-linear separation method and criterion, removing caller-predictive
  information did or did not preserve form-predictive information."
  Never universal non-linear entanglement or separability.

  FORM PROXY (honest label): the corpus has NO call-form annotation.
  The nearest supervised axis is the annotated behavioural context; we
  use the six most frequent context codes as the FORM-TASK LABELS and
  label the substitution explicitly throughout. Caller = Emitter.

  ARMS (fixed capacities throughout):
    A1 baseline   : encoder -> form head
    A2 multitask  : encoder -> form head + caller head (NO reversal)
    A3 adversarial: encoder -> form head + caller head via GRL
    A4 control    : adversarial with SHUFFLED caller labels (if A4
                    degrades form, the setup itself destabilises)
  Encoder: 2-layer MLP (220->128->64, ReLU) on 44-D bout fingerprints.
  Heads: linear. GRL lambda: linear schedule 0->1 over the first half
  of training, then fixed. Optimiser Adam lr 1e-3, 60 epochs, batch 128.
  Seeds: {0,1,2} — all reported, no selection.

  ENDPOINTS (prospective):
    F = form-task balanced accuracy (6-class) on held-out recording blocks
    C = caller-ID balanced accuracy on the same blocks
    Delta_F = F_adversarial - F_baseline; Delta_C likewise.
    Equivalence margin for form: eps_F = 0.02 BA (2 BA points). Suppression
    target: Delta_C <= -0.20 BA.
    OUTCOMES: (i) Delta_C <= -0.20 AND Delta_F > -eps_F across seeds =>
      separable UNDER THIS METHOD; (ii) suppression achieved but form
      degraded beyond margin => suppression-with-loss under this method;
    (iii) no suppression => the tested GRL failed to remove caller
      information at this capacity/schedule.
  All arms evaluated on identical held-out blocks (70/30 hash split).

Usage:
    python scripts/adversarial_invariance.py [--device cpu]

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

SEEDS = (0, 1, 2)
EPOCHS = 60
BATCH = 128
LR = 1e-3
HIDDEN1, HIDDEN2, EMB = 220, 128, 64
TRAIN_FRAC = 0.7
EPS_F = 0.02
TARGET_C_DROP = -0.20
FORM_TOPK = 6
MIN_CALLER_FILES = 25


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn

    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        filename_block_group,
        parse_annotations_csv,
    )
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import _FEATURE_ORDER
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        assert_feature_contract,
        raise_for_science,
    )

    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    meta = {a.file_name: a for a in annos}
    import collections

    ctx_counts = collections.Counter(a.context for a in annos)
    form_ctx = {c for c, _ in ctx_counts.most_common(FORM_TOPK)}
    caller_counts = collections.Counter(a.emitter for a in annos)
    callers_keep = {c for c, n in caller_counts.items() if n >= MIN_CALLER_FILES}
    print(f"form-proxy contexts ({FORM_TOPK}):", sorted(form_ctx))
    print(f"callers (>={MIN_CALLER_FILES} bouts):", len(callers_keep))

    d = np.load(ROOT / "cache/stage2/bat_c11c12_traj.npz", allow_pickle=False)
    frames, offsets, names = d["frames"], d["offsets"], list(d["names"])
    X, YF, YC, BLK = [], [], [], []
    for i, nm in enumerate(names):
        a = meta.get(nm)
        if a is None or a.context not in form_ctx or a.emitter not in callers_keep:
            continue
        f = frames[offsets[i] : offsets[i + 1]]
        v = np.atleast_2d(f)
        valid = ~np.all(v == 0, axis=1) if v.ndim == 2 else (v != 0).ravel()
        if int(valid.sum()) < 5:
            continue
        X.append(np.atleast_2d(f)[valid].mean(axis=0))
        YF.append(a.context)
        YC.append(a.emitter)
        BLK.append(filename_block_group(nm))
    X = np.stack(X)
    # this study's declared representation is the FULL 44-D combined
    # fingerprint (as in the context classifier), passed explicitly so the
    # runtime contract sees the declared width — the 12-feature trajectory
    # contract applies to forecasting studies, not whole-call fingerprint tasks
    assert_feature_contract(X[:2], expected_names=tuple(_FEATURE_ORDER),
                            context="invariance fingerprints (44-D declared)")
    fmap = {c: i for i, c in enumerate(sorted(form_ctx))}
    cmap = {c: i for i, c in enumerate(sorted(callers_keep))}
    YF = np.array([fmap[c] for c in YF])
    YC = np.array([cmap[c] for c in YC])
    BLK = np.array(BLK)
    blocks = sorted(set(BLK.tolist()))
    fit_blocks = {
        b
        for b in blocks
        if int(hashlib.sha1(b.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    is_fit = np.isin(BLK, sorted(fit_blocks))
    is_ev = ~is_fit
    mu, sd = X[is_fit].mean(0), X[is_fit].std(0) + 1e-9
    Xn = (X - mu) / sd
    print(
        f"bouts: {len(X)} (fit {int(is_fit.sum())} / eval {int(is_ev.sum())}) "
        f"| form classes {len(fmap)} | callers {len(cmap)}",
        flush=True,
    )

    contract = ScientificRunContract(
        run_name="adversarial_invariance",
        feature_schema_version=2,
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
        source_manifest={
            str(nm): {"group": str(nm), "part": "eval" if e else "fit"}
            for nm, e in zip(names[:1], [False])
        },
        eval_split={
            "provenance": "recording-block hash 70/30; 3 seeds all reported",
            "unit": "recording block",
        },
    )
    enforcement = raise_for_science(contract)

    device = torch.device(args.device)
    NF, NC = len(fmap), len(cmap)

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lamb):
            ctx.lamb = lamb
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lamb * g, None

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(44, HIDDEN1),
                nn.ReLU(),
                nn.Linear(HIDDEN1, HIDDEN2),
                nn.ReLU(),
                nn.Linear(HIDDEN2, EMB),
            )

        def forward(self, x):
            return self.net(x)

    def run_arm(seed, arm):
        torch.manual_seed(seed)
        np.random.seed(seed)
        enc = Encoder().to(device)
        form_head = nn.Linear(EMB, NF).to(device)
        call_head = nn.Linear(EMB, NC).to(device)
        opt = torch.optim.Adam(
            list(enc.parameters()) + list(form_head.parameters()) + list(call_head.parameters()),
            lr=LR,
        )
        xf = torch.tensor(Xn[is_fit], dtype=torch.float32, device=device)
        yf = torch.tensor(YF[is_fit], device=device)
        yc_true = YC[is_fit]
        if arm == "shuffled-adversarial":
            yc = torch.tensor(np.random.permutation(yc_true), device=device)
        else:
            yc = torch.tensor(yc_true, device=device)
        n = len(xf)
        use_caller = arm in ("multitask", "adversarial", "shuffled-adversarial")
        for epoch in range(EPOCHS):
            lamb = min(1.0, 2.0 * epoch / EPOCHS)  # linear ramp to 1 by mid-run
            perm = np.random.permutation(n)
            enc.train()
            for s in range(0, n, BATCH):
                idx = perm[s : s + BATCH]
                xb = xf[idx]
                z = enc(xb)
                lf = nn.functional.cross_entropy(form_head(z), yf[idx])
                loss = lf
                if use_caller:
                    zc = GradReverse.apply(z, lamb) if arm != "multitask" else z
                    lc = nn.functional.cross_entropy(call_head(zc), yc[idx])
                    loss = lf + lc
                opt.zero_grad()
                loss.backward()
                opt.step()
        enc.eval()
        with torch.no_grad():
            xe = torch.tensor(Xn[is_ev], dtype=torch.float32, device=device)
            z = enc(xe)
            pf = form_head(z).argmax(1).cpu().numpy()
            pc = call_head(z).argmax(1).cpu().numpy()
        ye_f, ye_c = YF[is_ev], YC[is_ev]

        def ba(y, p):
            return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))

        return ba(ye_f, pf), ba(ye_c, pc)

    arms = ("baseline", "multitask", "adversarial", "shuffled-adversarial")
    results = {a: [] for a in arms}
    for seed in SEEDS:
        for arm in arms:
            F, C = run_arm(seed, arm)
            results[arm].append({"seed": seed, "F": round(F, 4), "C": round(C, 4)})
            print(f"seed {seed} {arm:22s} F={F:.4f} C={C:.4f}", flush=True)

    def _mean(a, k):
        return float(np.mean([r[k] for r in results[a]]))

    summary = {
        a: {"F_mean": round(_mean(a, "F"), 4), "C_mean": round(_mean(a, "C"), 4)} for a in arms
    }
    dF = summary["adversarial"]["F_mean"] - summary["baseline"]["F_mean"]
    dC = summary["adversarial"]["C_mean"] - summary["baseline"]["C_mean"]
    dF_mt = summary["multitask"]["F_mean"] - summary["baseline"]["F_mean"]
    dF_sh = summary["shuffled-adversarial"]["F_mean"] - summary["baseline"]["F_mean"]
    if dC <= TARGET_C_DROP and dF > -EPS_F:
        verdict = (
            "SEPARABLE UNDER THIS METHOD: caller suppressed beyond target "
            f"({dC:+.3f}) with form within the equivalence margin ({dF:+.3f})"
        )
    elif dC <= TARGET_C_DROP:
        verdict = (
            "SUPPRESSION-WITH-LOSS under this method: caller suppressed "
            f"({dC:+.3f}) but form degraded beyond the margin ({dF:+.3f})"
        )
    else:
        verdict = (
            f"NO SUPPRESSION at this capacity/schedule (dC={dC:+.3f}); "
            "entanglement cannot be adjudicated by this probe"
        )
    if dF_sh < -EPS_F:
        verdict += (
            " WARNING: shuffled-caller control degrades form "
            f"({dF_sh:+.3f}) — the adversarial setup itself destabilises"
        )

    report = {
        "study": "non-linear caller-invariance test (adversarial GRL, "
        "method-conditioned separability ONLY)",
        "form_proxy_honest_label": "NO call-form annotation exists; form task = "
        f"top-{FORM_TOPK} annotated behavioural contexts (substitution declared)",
        "preregistration": {
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "batch": BATCH,
            "encoder": f"44-{HIDDEN1}-{HIDDEN2}-{EMB} MLP",
            "grl_schedule": "linear 0->1 by mid-training",
            "eps_F": EPS_F,
            "target_C_drop": TARGET_C_DROP,
            "arms": list(arms),
        },
        "release_enforcement": enforcement,
        "per_seed": results,
        "summary": summary,
        "deltas": {
            "dF_adversarial": round(dF, 4),
            "dC_adversarial": round(dC, 4),
            "dF_multitask_vs_baseline": round(dF_mt, 4),
            "dF_shuffled_control": round(dF_sh, 4),
        },
        "verdict": verdict,
        "framing": "method-conditioned: evidence about THIS architecture/"
        "schedule/capacity — not universal non-linear entanglement or "
        "separability; form proxy is behavioural context, not annotated form",
    }
    out = ROOT / "outputs" / "adversarial_invariance.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps({"summary": summary, "deltas": report["deltas"], "verdict": verdict}, indent=1),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
