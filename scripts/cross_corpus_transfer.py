#!/usr/bin/env python3
"""Cross-corpus MODEL-transfer matrix (frozen scaler + coefficients).

FRAMING (review-directed): this is NOT procedure portability (already
established by per-corpus refits). The question: does ONE fitted acoustic
dynamics map transfer across corpora WITHOUT refitting? Diagnostic
experiment only — a frozen cross-species model is never silently made a
production default.

THREE ARMS PER TARGET (identical eligible positions, same contract):
  persistence      trivial baseline
  target-refit     Ridge refit on the target's fit side (procedure-
                   portability optimum; alpha on target train-side val)
  source-frozen    scaler AND coefficients trained on the SOURCE corpus,
                   applied unchanged
PRIMARY: Delta_transfer = L(source-frozen) - L(target-refit).
OUTCOME ROWS (frozen): ~target-refit => portable learned local mapping;
> persistence but worse than refit => transferable dynamics + corpus-
specific calibration; ~or worse than persistence => procedure ports but
fitted dynamics are corpus-specific under these coordinates.
CAUTION (in the artifact): physical acoustic coordinates make frozen-
scaler transfer deliberately demanding — failure may be distribution
shift (amplitude, frequency range, hardware, vocal regime), NOT
species-specific dynamics.

PRESPECIFIED PANEL (fixed before running, no cherry-picking):
  sources: bat (the original learned map), chiffchaff (song-to-song)
  targets: chiffchaff, pipit, littleowl
  pairs run: bat->chiffchaff, bat->pipit, bat->littleowl,
             chiffchaff->pipit
  (marmoset: sparse long-history support; dolphin: within-recording
   scope — both EXCLUDED from the panel and labelled)

Usage:
    python scripts/cross_corpus_transfer.py

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

H_FRAMES = 4
LAGS = (4, 8, 12, 20)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
TRAIN_FRAC = 0.7
VAL_FRAC = 0.25
SEED = 0
PANEL = (("bat", "chiffchaff"), ("bat", "pipit"), ("bat", "littleowl"), ("chiffchaff", "pipit"))


def _extract_bat(bouts):
    import numpy as np

    X6, Y = [], []
    for b in bouts:
        z, v = b["z"], b["valid"]
        T = len(z)
        for t in range(max(LAGS) + 1, T - H_FRAMES):
            if v[t] and v[t - 1] and v[t + H_FRAMES] and all(v[t - k] for k in LAGS):
                X6.append(np.concatenate([z[t], z[t - 1]] + [z[t - k] for k in LAGS]))
                Y.append(z[t + H_FRAMES])
    return np.stack(X6), np.stack(Y)


def _load_avian(species):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from avian_forecasting_portability import SPECIES as AV_SPECIES, _extract
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )

    local = {"chiffchaff": {"dir": "chiffchaff_id",
                             "group_re": r"(chiffchaff-fg_cutted_day\d+_PC\d+)"}}
    sp = local.get(species) or AV_SPECIES[species]
    raw = ROOT / "data" / "alp" / sp["dir"] / "v0.1.0" / "raw"
    wavs = sorted((raw / "wav" / f"{species}-fg").glob("*.wav"))
    import re

    gre = re.compile(sp["group_re"])
    jobs = [(p, p.name, species) for p in wavs]
    with ProcessPoolExecutor(max_workers=24) as ex:
        results = [r for r in ex.map(_extract, jobs, chunksize=8) if r]
    fidx = [_NAME_TO_IDX[n] for n in supported_features()]
    clips = []
    for name, f44 in results:
        f = f44[:, fidx]
        valid = ~np.all(f == 0, axis=1)
        m = gre.search(name)
        group = m.group(1) if m else Path(name).stem
        if valid.sum() >= 40:
            clips.append({"name": name, "group": group, "z": f, "valid": valid})
    return clips


def _load_bat():
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
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
    audit = json.loads((ROOT / "outputs/history_support_audit.json").read_text())
    frozen = {
        (p["caller"], p["recipient"])
        for p in audit["support_audit"]["dyadic_cells_frozen"]["pairs"].values()
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
            f = frames[offsets[i] : offsets[i + 1]][:, fidx].astype(np.float64)
            valid = ~np.all(f == 0, axis=1)
            if valid.sum() >= 40:
                bouts.append(
                    {"name": nm, "group": filename_block_group(nm), "z": f, "valid": valid}
                )
    return bouts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    import numpy as np

    from acoustic_dynamics_report import ridge_fit, ridge_predict

    def _positions(units):
        X6, Y, GRP = [], [], []
        for b in units:
            z, v = b["z"], b["valid"]
            T = len(z)
            for t in range(max(LAGS) + 1, T - H_FRAMES):
                if v[t] and v[t - 1] and v[t + H_FRAMES] and all(v[t - k] for k in LAGS):
                    X6.append(np.concatenate([z[t], z[t - 1]] + [z[t - k] for k in LAGS]))
                    Y.append(z[t + H_FRAMES])
                    GRP.append(b["group"])
        return np.stack(X6), np.stack(Y), np.array(GRP)

    def _split(groups):
        gs = sorted(set(groups.tolist()))
        fit = {
            g
            for g in gs
            if int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
        }
        rng = np.random.RandomState(SEED)
        fl = sorted(fit)
        rng.shuffle(fl)
        val = set(fl[: max(1, int(len(fl) * VAL_FRAC))])
        fit_only = set(fl) - val
        return (
            np.isin(groups, sorted(fit_only)),
            np.isin(groups, sorted(val)),
            ~np.isin(groups, sorted(fit_only | val)),
        )

    def _loss(r):
        return float(np.sqrt((r**2).mean(0)).mean())

    def _fit_full(X, Y, is_fit, is_val):
        mu_x, sd_x = X[is_fit].mean(0), X[is_fit].std(0) + 1e-9
        mu_y, sd_y = Y[is_fit].mean(0), Y[is_fit].std(0) + 1e-9
        Xn = (X - mu_x) / sd_x
        Yn = (Y - mu_y) / sd_y
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(Xn[is_fit], Yn[is_fit], alpha)
            lv = _loss(ridge_predict(w, b, Xn[is_val]) - Yn[is_val])
            if lv < best[1]:
                best = (alpha, lv)
        w, b = ridge_fit(Xn[is_fit], Yn[is_fit], best[0])
        return {
            "mu_x": mu_x,
            "sd_x": sd_x,
            "mu_y": mu_y,
            "sd_y": sd_y,
            "w": w,
            "b": b,
            "alpha": best[0],
        }

    def _apply(model, X, Y):
        Xn = (X - model["mu_x"]) / model["sd_x"]
        r = ridge_predict(model["w"], model["b"], Xn) - (Y - model["mu_y"]) / model["sd_y"]
        return _loss(r)

    ROOT / "outputs"
    sources = {}
    for src in ("bat", "chiffchaff"):
        units = _load_bat() if src == "bat" else _load_avian(src)
        X, Y, G = _positions(units)
        is_f, is_v, _ = _split(G)
        sources[src] = _fit_full(X, Y, is_f, is_v)
        print(f"source {src}: {len(X)} positions, alpha {sources[src]['alpha']}", flush=True)

    results = {}
    for src, tgt in PANEL:
        units = _load_avian(tgt)
        X, Y, G = _positions(units)
        is_f, is_v, is_e = _split(G)
        tgt_refit = _fit_full(X, Y, is_f, is_v)
        L_refit = _apply(tgt_refit, X[is_e], Y[is_e])
        L_frozen = _apply(sources[src], X[is_e], Y[is_e])
        # persistence in TARGET-standardised units (target fit stats)
        mu_y, sd_y = Y[is_f].mean(0), Y[is_f].std(0) + 1e-9
        r_p = (Y[is_e] - mu_y) / sd_y - (X[is_e][:, :12] - mu_y) / sd_y
        L_pers = _loss(r_p)
        delta = L_frozen - L_refit
        results[f"{src}->{tgt}"] = {
            "L_persistence": round(L_pers, 4),
            "L_target_refit": round(L_refit, 4),
            "L_source_frozen": round(L_frozen, 4),
            "Delta_transfer": round(delta, 4),
            "n_eval": int(is_e.sum()),
            "target_refit_alpha": tgt_refit["alpha"],
        }
        print(
            f"{src}->{tgt}: pers {L_pers:.4f} refit {L_refit:.4f} "
            f"frozen {L_frozen:.4f} delta {delta:+.4f}",
            flush=True,
        )

    for k, v in results.items():
        if v["L_source_frozen"] <= v["L_target_refit"] * 1.05:
            verdict = "frozen ~ target-refit: portable learned local mapping"
        elif v["L_source_frozen"] < v["L_persistence"]:
            verdict = (
                "transferable dynamics + corpus-specific calibration "
                "(frozen beats persistence, loses to refit)"
            )
        else:
            verdict = (
                "procedure ports but fitted dynamics are corpus-"
                "specific under these coordinates (frozen ~ or worse "
                "than persistence)"
            )
        v["verdict"] = verdict

    report = {
        "study": "cross-corpus MODEL transfer (frozen scaler + coefficients; "
        "distinct from the established per-corpus PROCEDURE portability)",
        "panel": [list(p) for p in PANEL],
        "excluded": {
            "marmoset": "sparse long-history support",
            "dolphin": "within-recording scope",
        },
        "results": results,
        "caution": "physical acoustic coordinates make frozen-scaler transfer "
        "deliberately demanding: failure may be distribution shift "
        "(amplitude, frequency range, recording hardware, vocal regime), "
        "NOT species-specific dynamics",
        "firewall": "learned-weight transfer across species is a DIAGNOSTIC "
        "experiment; never a silent production default",
    }
    out = ROOT / "outputs" / "cross_corpus_transfer.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(results, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
