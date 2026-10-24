#!/usr/bin/env python3
"""Fresh-groups test of directional context specialisation (preregistered).

PREREGISTRATION (frozen before any evaluation; wording per review):

  MOTIVATING PRIOR (NOT truth): the occupancy-dynamics study found
  Gamma_11 = +0.0039 [-0.0013,+0.0091] inside the 1% equivalence margin
  while Gamma_12 = +0.0093 [+0.0053,+0.0129] exceeded it at the CI edge.

  HYPOTHESIS: transfer into C12 incurs a larger predictive penalty than
  transfer into C11 — C12 local dynamics are modestly MORE
  context-specialised under this representation.

  PRIMARY ENDPOINT: Delta_asym = Gamma_12 - Gamma_11 > 0, evaluated on
  COMMON ACOUSTIC SUPPORT (k=10 NN context mix), same 12-feature
  representation, 40 ms horizon, ridge family, training-fitted scaling,
  and transfer definitions as the motivating study — all frozen.

  MINIMUM MEANINGFUL DIFFERENCE (prospective, from design rationale):
  delta_min = 0.005 nRMSE — half the equivalence margin. Rationale: an
  asymmetry smaller than half the margin cannot alter the architectural
  interpretation (parallel G/F/R with shared F), whatever its sign; it
  would be statistical asymmetry without practical import. NOT chosen
  from the observed 0.0054.

  SECONDARY (distinct, no significance-difference fallacy): Gamma_11
  and Gamma_12 each vs the epsilon=0.010 equivalence margin.

  POPULATION (the freshness claim, honestly labelled): caller-recipient
  pairs NEVER analysed in any prior study of this programme — both
  contexts >=100 eligible positions, outside the original ten frozen
  pairs. Four such pairs exist (205-218, 207-208, 213-228, 215-208;
  2,629 positions total). These are PREVIOUSLY-UNANALYSED RECORDING
  GROUPS FROM THE SAME CORPUS — not newly collected sessions; a
  genuinely fresh-session replication remains open and this limitation
  is stated in the artifact. Models are trained on the SAME fit-side
  material as the motivating study (training reuse is deliberate: the
  hypothesis concerns the EVALUATION populations' transfer costs).

  OUTCOME TABLE (frozen): Delta_asym > 0 with CI above 0 AND above
  delta_min = meaningful C12 specialisation; > 0 CI above 0 but below
  delta_min = statistical asymmetry, weak practical import; ~0 with
  both directions equivalent = shared local dynamics supported; wide
  CI = insufficient; reversed = prior asymmetry unstable. Any positive
  result remains CONTEXT-SPECIFIC ACOUSTIC PREDICTIVE DYNAMICS — not
  syntax, meaning, or receiver use.

Usage:
    python scripts/asymmetry_fresh_groups.py

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

TRAIN_FRAC = 0.7
VAL_FRAC = 0.25
H_FRAMES = 4
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
BOOT_B = 2000
SEED = 0
K_NN = 10
EPSILON = 0.010
DELTA_MIN = 0.005
MIN_PAIR_POSITIONS = 100
NEW_PAIRS = ((205, 218), (207, 208), (213, 228), (215, 208))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    import numpy as np
    from scipy.spatial import cKDTree

    from acoustic_dynamics_report import ridge_fit, ridge_predict
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )
    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        filename_block_group,
        parse_annotations_csv,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        assert_feature_contract,
    )

    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    meta = {a.file_name: a for a in annos}
    audit = json.loads((ROOT / "outputs/history_support_audit.json").read_text())
    orig_pairs = {
        (p["caller"], p["recipient"])
        for p in audit["support_audit"]["dyadic_cells_frozen"]["pairs"].values()
    }
    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]

    # ---- training positions: the ORIGINAL frozen population's fit side
    # (same as the motivating study); evaluation: NEW pairs only
    train_pos, new_pos = [], []
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
            f = frames[offsets[i] : offsets[i + 1]][:, fidx].astype(np.float64)
            valid = ~np.all(f == 0, axis=1)
            is_orig = (a.emitter, a.addressee) in orig_pairs
            is_new = (a.emitter, a.addressee) in NEW_PAIRS
            T = len(f)
            for t in range(1, T - H_FRAMES):
                if not (valid[t] and valid[t + H_FRAMES]):
                    continue
                rec = (f[t], f[t + H_FRAMES], a.context, filename_block_group(nm))
                if is_orig:
                    train_pos.append(rec)
                elif is_new:
                    new_pos.append(rec)
    S_tr = np.stack([p[0] for p in train_pos])
    Y_tr = np.stack([p[1] for p in train_pos])
    C_tr = np.array([p[2] for p in train_pos])
    BLK_tr = np.array([p[3] for p in train_pos])
    S_new = np.stack([p[0] for p in new_pos])
    Y_new = np.stack([p[1] for p in new_pos])
    C_new = np.array([p[2] for p in new_pos])
    BLK_new = np.array([p[3] for p in new_pos])
    print(
        f"train (original pop): {len(S_tr)} | NEW-pair eval: {len(S_new)} "
        f"(C11 {int((C_new == 11).sum())} / C12 {int((C_new == 12).sum())})",
        flush=True,
    )
    assert_feature_contract(S_tr, context="asymmetry fitting")

    blocks = sorted(set(BLK_tr.tolist()))
    fit_blocks = {
        b
        for b in blocks
        if int(hashlib.sha1(b.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    rng = np.random.RandomState(SEED)
    fit_list = sorted(fit_blocks)
    rng.shuffle(fit_list)
    val_blocks = set(fit_list[: max(1, int(len(fit_list) * VAL_FRAC))])
    fit_blocks = set(fit_list) - val_blocks
    is_fit = np.isin(BLK_tr, sorted(fit_blocks))
    is_val = np.isin(BLK_tr, sorted(val_blocks))

    mu_s, sd_s = S_tr[is_fit].mean(0), S_tr[is_fit].std(0) + 1e-9
    mu_y, sd_y = Y_tr[is_fit].mean(0), Y_tr[is_fit].std(0) + 1e-9
    Sn_tr = (S_tr - mu_s) / sd_s
    Yn_tr = (Y_tr - mu_y) / sd_y
    Sn_new = (S_new - mu_s) / sd_s
    Yn_new = (Y_new - mu_y) / sd_y

    # common support on the NEW pairs (training-tree NN context mix)
    tree = cKDTree(Sn_tr[is_fit])
    Cfit = C_tr[is_fit]
    _, nn = tree.query(Sn_new, k=K_NN)
    nn_c = Cfit[nn]
    common = (nn_c == 11).any(1) & (nn_c == 12).any(1)
    ev = np.where(common)[0]
    print(
        f"common-support NEW positions: {int(common.sum())}/{len(Sn_new)} "
        f"({100 * float(common.mean()):.1f}%)",
        flush=True,
    )

    # models (identical family; alpha on train-side val)
    def _pick_alpha(mask):
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(Sn_tr[mask], Yn_tr[mask], alpha)
            lv = float(
                np.sqrt(((ridge_predict(w, b, Sn_tr[is_val]) - Yn_tr[is_val]) ** 2).mean(0)).mean()
            )
            if lv < best[1]:
                best = (alpha, lv)
        return best[0]

    models = {}
    for name, mask in (("F11", is_fit & (C_tr == 11)), ("F12", is_fit & (C_tr == 12))):
        alpha = _pick_alpha(mask)
        w, b = ridge_fit(Sn_tr[mask], Yn_tr[mask], alpha)
        models[name] = (w, b, alpha)

    def _loss(name, idx):
        w, b, _ = models[name]
        r = ridge_predict(w, b, Sn_new[idx]) - Yn_new[idx]
        return float(np.sqrt((r**2).mean(0)).mean())

    m11 = ev[C_new[ev] == 11]
    m12 = ev[C_new[ev] == 12]
    L = {
        "L_11->11": _loss("F11", m11),
        "L_12->11": _loss("F12", m11),
        "L_11->12": _loss("F11", m12),
        "L_12->12": _loss("F12", m12),
    }
    g11 = L["L_12->11"] - L["L_11->11"]
    g12 = L["L_11->12"] - L["L_12->12"]
    asym = g12 - g11

    # bootstrap over the NEW pairs' recording blocks
    by_blk: dict = {}
    for i in ev:
        by_blk.setdefault(BLK_new[i], []).append(i)
    keys = sorted(by_blk)
    brng = np.random.RandomState(SEED)

    def ll(r):
        return float(np.sqrt((r**2).mean(0)).mean())

    a_reps, g11_reps, g12_reps = [], [], []
    for _ in range(BOOT_B):
        pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
        idx = np.concatenate([by_blk[g] for g in pick])
        i11 = idx[C_new[idx] == 11]
        i12 = idx[C_new[idx] == 12]
        if len(i11) < 5 or len(i12) < 5:
            continue
        w11, b11, _ = models["F11"]
        w12, b12, _ = models["F12"]
        r1111 = ridge_predict(w11, b11, Sn_new[i11]) - Yn_new[i11]
        r1211 = ridge_predict(w12, b12, Sn_new[i11]) - Yn_new[i11]
        r1112 = ridge_predict(w11, b11, Sn_new[i12]) - Yn_new[i12]
        r1212 = ridge_predict(w12, b12, Sn_new[i12]) - Yn_new[i12]
        _g11 = ll(r1211) - ll(r1111)
        _g12 = ll(r1112) - ll(r1212)
        g11_reps.append(_g11)
        g12_reps.append(_g12)
        a_reps.append(_g12 - _g11)
    ci_a = [float(np.percentile(a_reps, 2.5)), float(np.percentile(a_reps, 97.5))]
    ci_11 = [float(np.percentile(g11_reps, 2.5)), float(np.percentile(g11_reps, 97.5))]
    ci_12 = [float(np.percentile(g12_reps, 2.5)), float(np.percentile(g12_reps, 97.5))]

    pos_ci = ci_a[0] > 0
    meaningful = ci_a[0] > DELTA_MIN
    eq11 = ci_11[0] > -EPSILON and ci_11[1] < EPSILON
    eq12 = ci_12[0] > -EPSILON and ci_12[1] < EPSILON
    if pos_ci and meaningful:
        verdict = "MEANINGFUL C12 specialisation: Delta_asym CI above 0 AND above delta_min"
    elif pos_ci:
        verdict = (
            "statistical asymmetry with weak practical import (CI above 0 but below delta_min)"
        )
    elif abs(float(np.mean(a_reps))) < 1e-9 and eq11 and eq12:
        verdict = "shared local dynamics supported (both directions equivalent)"
    elif ci_a[1] - ci_a[0] > 0.02:
        verdict = "wide CI — fresh groups insufficient to resolve"
    elif ci_a[1] <= 0:
        verdict = "REVERSED — prior asymmetry not stable on fresh groups"
    else:
        verdict = "unresolved (CI spans 0 and delta_min)"

    report = {
        "study": "fresh-groups directional specialisation test "
        "(preregistered; see module docstring)",
        "population_note": "four caller-recipient pairs NEVER analysed "
        "before in this programme (both contexts >=100 positions, outside "
        "the original ten); PREVIOUSLY-UNANALYSED RECORDING GROUPS FROM "
        "THE SAME CORPUS — not newly collected sessions; a fresh-session "
        "replication remains open",
        "primary": {
            "Delta_asym = Gamma_12 - Gamma_11": round(float(asym), 4),
            "ci95": [round(x, 4) for x in ci_a],
            "delta_min": DELTA_MIN,
            "delta_min_rationale": "half the equivalence margin: an "
            "asymmetry below this cannot alter the architectural "
            "interpretation whatever its sign; NOT derived from the "
            "observed 0.0054",
            "positive_ci_excludes_0": bool(pos_ci),
            "meaningful": bool(meaningful),
        },
        "secondary": {
            "Gamma_11": round(float(g11), 4),
            "ci11": [round(x, 4) for x in ci_11],
            "equivalent_11": bool(eq11),
            "Gamma_12": round(float(g12), 4),
            "ci12": [round(x, 4) for x in ci_12],
            "equivalent_12": bool(eq12),
            "epsilon": EPSILON,
        },
        "transfer_matrix": {k: round(v, 4) for k, v in L.items()},
        "counts": {
            "train_positions": int(len(S_tr)),
            "new_eval_total": int(len(S_new)),
            "new_eval_common_support": int(len(ev)),
            "new_pairs": len(NEW_PAIRS),
            "common_support_fraction": round(float(common.mean()), 3),
        },
        "verdict": verdict,
        "framing": "any positive result is CONTEXT-SPECIFIC ACOUSTIC "
        "PREDICTIVE DYNAMICS — not syntax, meaning, or receiver use",
    }
    out = ROOT / "outputs" / "asymmetry_fresh_groups.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                "Delta_asym": round(float(asym), 4),
                "ci": ci_a,
                "Gamma_11": round(float(g11), 4),
                "ci11": ci_11,
                "Gamma_12": round(float(g12), 4),
                "ci12": ci_12,
                "verdict": verdict,
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
