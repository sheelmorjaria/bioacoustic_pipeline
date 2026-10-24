#!/usr/bin/env python3
"""Longer-history and context-dependent acoustic forecasting (frozen spec).

SPEC (frozen in outputs/history_support_audit.json BEFORE this run):
representation and 40 ms horizon FIXED; older history u = [s_{t-40ms},
s_{t-80ms}, s_{t-120ms}, s_{t-200ms}]; recent x = [s_t, s_{t-10ms}];
context code c in {-1,+1}; five-model nested ladder; ONE primary contrast

    Delta_context-history = L(M_context-recent) - L(M_context-older-history)

(POSITIVE = context-dependent older-history terms improve held-out
prediction beyond context-level and recent-state terms), supporting

    Delta_older-history = L(M_short) - L(M_shared-longer).

IDENTICAL evaluation positions across arms (common eligible set; gaps
never compressed; estimand conditional on >=280 ms contiguous support).
ONE common training-fitted scaler across contexts and arms; context code
appended unscaled (+-1); per-arm alpha from the same grid on train-side
validation; recording-group (block) clustered paired bootstrap shared
across arms. Population = the audit's frozen dyadic pairs (equal
common-recipient weighting, both contexts). INTERPRETATION LIMITS: this
is label-conditioned acoustic prediction — not an audio-only context
classifier, not future-behaviour forecasting, not grammar/units/meaning;
history-helps and order-matters are not interchangeable; a
context-dependent linear mapping may approximate regions of a shared
nonlinear relationship.

STEP 4 (this script, --self-test): the four DECLARED synthetic fixtures
through the COMPLETE pipeline (eligibility, scaling, ladder, scoring):
  F1 shared dynamics + context means -> Delta_context-history ~ 0;
  F2 context-dependent RECENT effects, shared older -> Delta_context-
     history ~ 0 (the recent control must absorb; naive level-only would
     mislead);
  F3 known context-x-older effect -> Delta_context-history > 0;
  F4 gaps/padding -> positions crossing gaps excluded; appended padding
     never changes valid results.

Usage:
    python scripts/history_context_forecasting.py --self-test   # step 4
    python scripts/history_context_forecasting.py               # step 5 (ONE run)

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

H_MS = 4  # target frame offset (40 ms)
LAGS = (4, 8, 12, 20)  # older-history frame offsets (40/80/120/200 ms)
REQUIRED_FRAMES = 28
TRAIN_FRAC = 0.7
VAL_FRAC = 0.25
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
BOOT_B = 2000
SEED = 0
MIN_PAIR_POSITIONS = 200

MODELS = ("short", "shared", "level", "recent", "older")


# ---------------------------------------------------------------- pipeline


def _design(s, t):
    """x, u, y, c from a trajectory dict at position t (frame indices)."""
    import numpy as np

    x = np.concatenate([s["z"][t], s["z"][t - 1]])
    u = np.concatenate([s["z"][t - k] for k in LAGS])
    y = s["z"][t + H_MS]
    c = 1.0 if s["context"] == 12 else -1.0
    return x, u, y, c


def _arm_matrix(arm, X, U, C):
    """Nested ladder designs. Context code appended UNSCALED (+-1)."""
    import numpy as np

    ones = C[:, None]
    if arm == "short":
        return X
    base = np.hstack([X, U])
    if arm == "shared":
        return base
    if arm == "level":
        return np.hstack([base, ones])
    if arm == "recent":
        return np.hstack([base, ones, X * ones])
    if arm == "older":
        return np.hstack([base, ones, X * ones, U * ones])
    raise ValueError(arm)


def eligible_positions(track_valid):
    """Positions t with frames [t-20, t+4] inside ONE contiguous span."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import contiguous_spans

    spans = contiguous_spans(np.asarray(track_valid, bool))
    out = []
    for sid in sorted(set(spans.tolist())):
        if sid < 0:
            continue
        idx = np.where(spans == sid)[0]
        if len(idx) >= REQUIRED_FRAMES:
            lo, hi = int(idx[0]), int(idx[-1])
            out.extend(range(lo + max(LAGS), hi - H_MS + 1))
    return out


def run_ladder(rows, boot_b=0):
    """Fit the ladder on rows; return eval losses + paired bootstrap deltas.

    rows: dict with z (per-frame features), context, group per element.
    A row's design comes from ONE bout trajectory + position.
    """
    import numpy as np

    from acoustic_dynamics_report import ridge_fit, ridge_predict

    groups = sorted({r["group"] for r in rows})
    fit_blocks = {
        g
        for g in groups
        if int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    ev_groups = sorted(set(groups) - fit_blocks)
    # train-side validation split (by block) for alpha selection
    rng = np.random.RandomState(SEED)
    fit_list = sorted(fit_blocks)
    rng.shuffle(fit_list)
    n_val = max(1, int(len(fit_list) * VAL_FRAC))
    val_blocks = set(fit_list[:n_val])
    fit_blocks = fit_blocks - val_blocks

    tr = [r for r in rows if r["group"] in fit_blocks]
    va = [r for r in rows if r["group"] in val_blocks]
    ev = [r for r in rows if r["group"] in ev_groups]
    if not tr or not va or not ev:
        raise ValueError("empty split")

    def _mats(rs):
        X = np.stack([r["x"] for r in rs])
        U = np.stack([r["u"] for r in rs])
        Y = np.stack([r["y"] for r in rs])
        C = np.array([r["c"] for r in rs])
        return X, U, Y, C

    Xtr, Utr, Ytr, Ctr = _mats(tr)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-9  # common scaler, both contexts
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-9
    mu_u, sd_u = Utr.mean(0), Utr.std(0) + 1e-9
    Xtr = (Xtr - mu_x) / sd_x
    Utr = (Utr - mu_u) / sd_u
    Xev, Uev, Yev, Cev = _mats(ev)
    Xev = (Xev - mu_x) / sd_x
    Uev = (Uev - mu_u) / sd_u
    Yn = (Ytr - mu) / sd
    Yn_ev = (Yev - mu) / sd

    models = {}
    for arm in MODELS:
        A_tr = _arm_matrix(arm, Xtr, Utr, Ctr)
        Xva = (np.stack([r["x"] for r in va]) - mu_x) / sd_x
        Uva = (np.stack([r["u"] for r in va]) - mu_u) / sd_u
        Cva = np.array([r["c"] for r in va])
        Yva = (np.stack([r["y"] for r in va]) - mu) / sd
        A_va = _arm_matrix(arm, Xva, Uva, Cva)
        best = (None, np.inf)
        for alpha in ALPHAS:
            w, b = ridge_fit(A_tr, Yn, alpha)
            pred = ridge_predict(w, b, A_va)
            loss = float(np.sqrt(((pred - Yva) ** 2).mean(axis=0)).mean())
            if loss < best[1]:
                best = (alpha, loss)
        w, b = ridge_fit(A_tr, Yn, best[0])
        A_ev = _arm_matrix(arm, Xev, Uev, Cev)
        pred = ridge_predict(w, b, A_ev)
        resid = pred - Yn_ev
        models[arm] = {
            "alpha": best[0],
            "loss": float(np.sqrt((resid**2).mean(axis=0)).mean()),
            "_resid": resid,
        }

    def _loss(resid):
        import numpy as np

        return float(np.sqrt((resid**2).mean(axis=0)).mean())

    deltas = {
        "primary_delta_context_history": models["recent"]["loss"] - models["older"]["loss"],
        "supporting_delta_older_history": models["short"]["loss"] - models["shared"]["loss"],
    }
    ci = {}
    if boot_b:
        brng = np.random.RandomState(SEED)
        by_group: dict = {}
        for i, r in enumerate([e for e in ev]):
            by_group.setdefault(r["group"], []).append(i)
        keys = sorted(by_group)
        d_prim, d_old = [], []
        for _ in range(boot_b):
            pick = [keys[i] for i in brng.randint(0, len(keys), len(keys))]
            idx = np.concatenate([by_group[g] for g in pick])
            d_prim.append(
                _loss(models["recent"]["_resid"][idx]) - _loss(models["older"]["_resid"][idx])
            )
            d_old.append(
                _loss(models["short"]["_resid"][idx]) - _loss(models["shared"]["_resid"][idx])
            )
        for name, ds in (
            ("primary_delta_context_history", d_prim),
            ("supporting_delta_older_history", d_old),
        ):
            ci[name] = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
    for arm in MODELS:
        models[arm].pop("_resid", None)
    return {
        "n_fit": len(tr),
        "n_val": len(va),
        "n_eval": len(ev),
        "models": models,
        "deltas": deltas,
        "bootstrap_ci": ci,
    }


# ---------------------------------------------------------------- fixtures


def _fixture(generate, n_bouts=60, n_frames=90, seed=0):
    """Build synthetic bouts, run the ladder, return deltas.

    The generator receives the TRUE context code of the bout (+1/-1,
    matching _design's mapping), so planted context effects are visible
    to the model through the code it actually sees."""
    import numpy as np

    rng = np.random.RandomState(seed)
    rows = []
    for b in range(n_bouts):
        ctx = 12 if b % 2 else 11
        c_code = 1.0 if ctx == 12 else -1.0
        z = generate(rng, n_frames, c_code)
        valid = np.ones(n_frames, bool)
        track = {"z": z, "context": ctx, "group": f"g{b % 12}"}
        for t in eligible_positions(valid):
            x, u, y, c = _design(track, t)
            rows.append({"x": x, "u": u, "y": y, "c": c, "group": track["group"]})
    return run_ladder(rows, boot_b=0)


def _self_test() -> None:
    import numpy as np

    D = 3  # small feature dim for fixtures (pipeline-identical machinery)

    # F1: shared dynamics + context means (NO interactions)
    def f1(rng, n, c):
        z = np.zeros((n, D))
        z[0] = rng.randn(D)
        for t in range(1, n):
            z[t] = 0.7 * z[t - 1] + 0.15 * rng.randn(D)
        z += 1.5 * c  # context mean (shared dynamics)
        return z

    d1 = _fixture(f1, seed=1)

    # F2: context-dependent RECENT effects, shared older history
    def f2(rng, n, c):
        z = np.zeros((n, D))
        z[0] = rng.randn(D)
        for t in range(1, n):
            base = 0.6 * z[t - 1]
            recent = (0.25 if c > 0 else -0.25) * z[t - 1]
            z[t] = base + recent + 0.12 * rng.randn(D)
        return z

    d2 = _fixture(f2, seed=2)

    # F3: known context-x-OLDER effect, planted on NON-OVERLAPPING
    # positions so the recent features stay uninformative: each position's
    # frames [t-20, t+4] are exclusive to it; y depends on z[t-8] via c
    def f3_rows(rng, n_bouts=60, n_frames=90):
        rows = []
        for b in range(n_bouts):
            ctx = 12 if b % 2 else 11
            c = 1.0 if ctx == 12 else -1.0
            z = rng.randn(n_frames, D)
            group = f"g{b % 12}"
            for t in range(max(LAGS), n_frames - H_MS, 25):
                z[t + H_MS] = 0.8 * c * z[t - 8] + 0.3 * rng.randn(D)
                track = {"z": z, "context": ctx}
                for tt in (t,):  # single position per block
                    x, u, y, cc = _design(track, tt)
                    # y was overwritten above for THIS t only
                    rows.append({"x": x, "u": u, "y": z[tt + H_MS], "c": cc, "group": group})
        return rows

    rng3 = np.random.RandomState(3)
    d3 = run_ladder(f3_rows(rng3), boot_b=0)
    print(
        f"F1 shared+means:    primary {d1['deltas']['primary_delta_context_history']:+.4f} "
        f"(want ~0), older-history {d1['deltas']['supporting_delta_older_history']:+.4f}",
        flush=True,
    )
    print(
        f"F2 context-recent: primary {d2['deltas']['primary_delta_context_history']:+.4f} "
        f"(want ~0: recent control absorbs)",
        flush=True,
    )
    print(
        f"F3 ctx-x-older:    primary {d3['deltas']['primary_delta_context_history']:+.4f} "
        f"(want > 0)",
        flush=True,
    )
    assert abs(d1["deltas"]["primary_delta_context_history"]) < 0.02, d1
    assert abs(d2["deltas"]["primary_delta_context_history"]) < 0.02, d2
    assert d3["deltas"]["primary_delta_context_history"] > 0.02, d3

    # F4: gaps/padding — positions crossing a gap are excluded; appended
    # padding never changes valid results
    valid = np.ones(90, bool)
    valid[30:60] = False
    pos = eligible_positions(valid)
    assert all(
        not (30 <= t - LAGS[-1] or 30 <= t + H_MS <= 59) or (t - LAGS[-1] >= 60 or t + H_MS < 30)
        for t in pos
    )
    n_before = len(pos)
    padded = np.concatenate([valid, np.zeros(10, bool)])
    assert len(eligible_positions(padded)) == n_before  # padding inert
    print(f"F4 gaps/padding: {n_before} positions exclude gap-crossers; padding inert", flush=True)
    print("SELF-TEST OK (all four declared fixtures)", flush=True)


# ---------------------------------------------------------------- step 5


def _real_run() -> None:
    import numpy as np

    from bioacoustic_pipeline.ingestion.labeled_corpus import (
        filename_block_group,
        parse_annotations_csv,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )

    annos = [
        a
        for a in parse_annotations_csv(ROOT / "data/egyptian_fruit_bats/annotations.csv")
        if a.emitter > 0
    ]
    meta = {a.file_name: a for a in annos}
    audit = json.loads((ROOT / "outputs/history_support_audit.json").read_text())
    frozen = audit["support_audit"]["dyadic_cells_frozen"]["pairs"]
    frozen_pairs = {(p["caller"], p["recipient"]) for p in frozen.values()}

    seen: set = set()
    bouts = []
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
            if (a.emitter, a.addressee) not in frozen_pairs:
                continue
            f = frames[offsets[i] : offsets[i + 1]]
            valid = ~np.all(f == 0, axis=1)
            pos = eligible_positions(valid)
            if not pos:
                continue
            bouts.append(
                {
                    "name": nm,
                    "context": a.context,
                    "caller": a.emitter,
                    "recipient": a.addressee,
                    "group": filename_block_group(nm),
                    "frames": f,
                    "valid": valid,
                    "pos": pos,
                }
            )
    # shared scaler statistics need pooled features: fit-side determined
    # inside run_ladder via rows; rows built here
    rows = []
    for b in bouts:
        for t in b["pos"]:
            x, u, y, c = _design({"z": b["frames"], "context": b["context"]}, t)
            rows.append(
                {
                    "x": x,
                    "u": u,
                    "y": y,
                    "c": c,
                    "group": b["group"],
                    "caller": b["caller"],
                    "context": b["context"],
                }
            )
    files = sorted({b["name"] for b in bouts})
    contract = ScientificRunContract(
        run_name="history_context_forecasting",
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
                produced_by=f"contiguous spans >= {REQUIRED_FRAMES} frames "
                "(280 ms incl. history+window+horizon)",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(
            files, ["frozen-population"] * len(files), [filename_block_group(f) for f in files]
        ),
        eval_split={
            "provenance": "recording-block hash 70/30 within the "
            "audit's frozen dyadic population; train-side VAL 25% "
            "for alpha",
            "unit": "recording block",
        },
    )
    enforcement = raise_for_science(contract)
    print(
        f"population: {len(bouts)} bouts / {len(rows)} positions from "
        f"{len(frozen_pairs)} frozen pairs",
        flush=True,
    )

    out = run_ladder(rows, boot_b=BOOT_B)
    out.update(
        {
            "release_enforcement": enforcement,
            "population": {
                "n_bouts": len(bouts),
                "n_positions": len(rows),
                "frozen_pairs": len(frozen_pairs),
            },
            "spec": audit["frozen_study_spec"],
            "framing": "label-conditioned acoustic prediction — not an "
            "audio-only context classifier, not future-behaviour forecasting, "
            "not grammar/units/meaning",
        }
    )
    path = ROOT / "outputs" / "history_context_forecasting.json"
    path.write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                "models": {a: out["models"][a]["loss"] for a in MODELS},
                "deltas": out["deltas"],
                "bootstrap_ci": out["bootstrap_ci"],
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    else:
        _real_run()


if __name__ == "__main__":
    main()
