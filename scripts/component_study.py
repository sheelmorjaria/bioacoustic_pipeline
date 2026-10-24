#!/usr/bin/env python3
"""Component-specific information and recombination within a single call.

PREREGISTRATION (frozen before any association was computed):

  HYPOTHESIS (advisor-worded): the selected vocalisation class contains
  reusable internal acoustic components whose contributions combine
  systematically. Segmentation, prediction and acoustic similarity alone
  will NOT be interpreted as evidence of words, sentences, or meaning.

  PARENT VOCAL EVENT: a contiguous fingerprint-valid span of >= MIN_SPAN
  frames (40 ms / 10 ms grid) inside one source file — bounded by gated
  frames or file edges, with caller/recipient/context joined at file level
  (bout-level scope caveat documented). Parent boundaries are validity
  events, NOT chosen to produce attractive component sequences. Artificial
  padding and gate-starved windows are excluded by construction.

  CANDIDATE COMPONENTS: each parent split into 3 equal subregions (thirds —
  a declared MODELLING CHOICE and the onset/body/offset baseline the
  organised account must beat, not evidence of grammar). Each third is
  described by its mean fingerprint vector; a frozen vocabulary
  (FrozenDUSTED, k=K_VOCAB) is fit on TRAIN blocks only and assigns every
  third a neutral component class.

  TARGETS: primary = production context (C11 vs C12, the audited contrast);
  secondary = recipient identity within multi-recipient callers
  (multiplicity-flagged).

  MODELS (identical held-out calls):
    social    one-hot(emitter, recipient)                — no acoustics
    whole     ridge on the z-scored whole-call mean fingerprint
    unordered sorted-triple one-hot + duration           — set, no order
    position  per-position one-hots                      — where, no relation
    ordered   full-triple one-hot                        — arrangement

  PRIMARY CONTRAST: Delta_organisation = L(unordered) - L(ordered),
  L = 1 - balanced accuracy; recording-block-clustered paired bootstrap.

  COMBINATION HOLDOUT: distinct (prefix, suffix) component pairs are
  deterministically withheld from ALL training; prediction on those calls
  is compared against a whole-call template baseline (1-NN on the
  whole-call fingerprint). Generalisation above templates on withheld
  combinations is the computational test distinguishing a component system
  from memorised calls.

  REUSE TEST: independence of prefix and suffix component classes across
  calls (permutation chi-square) — components must recur in DIFFERENT
  combinations (A1B1, A1B2, A2B1, A2B2), not merely reappear.

  INTERPRETATION LADDER (prespecified): recurrent components only
  -> intra-call acoustic organisation; components predict different
  variables -> multiple information-bearing regions; organised model beats
  unordered on held-out -> organisation-associated information;
  (receiver tiers require the separate behavioural assay — see
  RECEIVER_HANDOFF.md).

Usage:
    python scripts/component_study.py --self-test
    python scripts/component_study.py

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

MIN_SPAN = 6  # frames; >= 6 so each third has >= 2 frames
K_VOCAB = 6
MIN_CALLS_PER_CELL = 6
BOOT_B = 2000
PERM_B = 2000
SEED = 0
TRAIN_FRAC = 0.7


def prereg_spec() -> dict:
    return {
        "title": "Component-specific information and recombination within a "
        "single animal call (computational portion)",
        "hypothesis": "reusable internal acoustic components whose "
        "contributions combine systematically; segmentation/prediction/"
        "similarity are NOT word, sentence or meaning evidence",
        "parent_event": f"contiguous fingerprint-valid span >= {MIN_SPAN} frames",
        "components": f"equal thirds; frozen vocab k={K_VOCAB} fit on TRAIN only",
        "targets": {
            "primary": "context C11 vs C12",
            "secondary": "recipient (multiplicity-flagged)",
        },
        "models": ["social", "whole", "unordered", "position", "ordered"],
        "primary_contrast": "Delta_organisation = L(unordered) - L(ordered), "
        "L = 1 - balanced accuracy, block-clustered paired bootstrap",
        "combination_holdout": "deterministic 1-in-4 of distinct (prefix,"
        "suffix) pairs withheld from ALL training; vs whole-call 1-NN "
        "template",
        "ladder": [
            "recurrent components -> intra-call acoustic organisation",
            "components predict different variables -> multiple information-bearing regions",
            "organised model beats unordered held-out -> organisation-associated information",
            "receiver tiers require the separate behavioural assay",
        ],
    }


# ------------------------------------------------------------------ inventory


def parent_events_from_caches() -> list[dict]:
    """Parent events = contiguous valid spans, from both frozen caches."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import contiguous_spans
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
    events = []
    seen_files: set = set()
    caches = [
        ROOT / "cache/stage2/bat_c11c12_traj.npz",
        ROOT / "cache/stage2/confirm_dyn_cv_traj.npz",
    ]
    for c in caches:
        if not c.exists():
            continue
        d = np.load(c, allow_pickle=False)
        frames, offsets = d["frames"], d["offsets"]
        names = list(d["names"])
        for i, nm in enumerate(names):
            if nm in seen_files or nm not in meta:
                continue
            seen_files.add(nm)
            a = meta[nm]
            if a.context not in (11, 12) or a.addressee <= 0:
                continue
            f = frames[offsets[i] : offsets[i + 1]]
            valid = ~np.all(f == 0, axis=1)
            spans = contiguous_spans(valid)
            for sid in sorted(set(spans.tolist())):
                if sid < 0:
                    continue
                idx = np.where(spans == sid)[0]
                if len(idx) < MIN_SPAN:
                    continue
                events.append(
                    {
                        "file": nm,
                        "caller": a.emitter,
                        "recipient": a.addressee,
                        "context": a.context,
                        "block": filename_block_group(nm),
                        "start": int(idx[0]),
                        "end": int(idx[-1]) + 1,
                        "frames": f[idx].astype(np.float64),
                    }
                )
    return events


def _thirds_mean(frames):
    import numpy as np

    n = len(frames)
    bounds = [round(n * k / 3) for k in range(4)]
    return np.stack([frames[bounds[i] : bounds[i + 1]].mean(axis=0) for i in range(3)])


def block_split(events) -> tuple[list, list]:
    """Recording-block split: all events of one block stay together."""
    blocks = sorted({e["block"] for e in events})
    train_blocks = {
        b
        for b in blocks
        if int(hashlib.sha1(b.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
    }
    tr = [e for e in events if e["block"] in train_blocks]
    ev = [e for e in events if e["block"] not in train_blocks]
    return tr, ev


# ------------------------------------------------------------------- models


def _ridge_fit(x, y_onehot, alpha=1.0):
    import numpy as np

    d = x.shape[1]
    x1 = np.hstack([x, np.ones((len(x), 1))])
    a = x1.T @ x1 + alpha * np.eye(d + 1)
    a[d, d] -= alpha  # unregularized intercept
    return np.linalg.solve(a, x1.T @ y_onehot)


def _ridge_predict(w, x):
    import numpy as np

    return np.hstack([x, np.ones((len(x), 1))]) @ w


def _balanced_acc(y_true, y_pred) -> float:
    import numpy as np

    classes = np.unique(y_true)
    accs = [(y_true[y_true == c] == y_pred[y_true == c]).mean() for c in classes]
    return float(np.mean(accs))


def _fit_predict_suite(train, test, feature_of) -> dict:
    """Fit all five models on train rows; predict test rows.

    ``feature_of(events) -> (X, y)`` builds the design for a model name.
    """
    import numpy as np

    out = {}
    y_tr = np.array([e["context"] for e in train])
    y_ev = np.array([e["context"] for e in test])
    classes = sorted(set(y_tr.tolist()) | set(y_ev.tolist()))

    def oh(y):
        return np.stack([(y == c).astype(float) for c in classes], axis=1)

    for name in ("social", "whole", "unordered", "position", "ordered"):
        xtr, xev = feature_of(name, train, test)
        if xtr.shape[1] == 0:
            out[name] = None
            continue
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
        w = _ridge_fit((xtr - mu) / sd, oh(y_tr))
        scores = _ridge_predict(w, (xev - mu) / sd)
        pred = np.array([classes[i] for i in scores.argmax(1)])
        out[name] = {
            "balanced_acc": _balanced_acc(y_ev, pred),
            "L": 1.0 - _balanced_acc(y_ev, pred),
            "n": len(test),
        }
    return out


def _design(name, train, test, comp_of, whole_of):
    """One-hot designs on component classes + social/whole baselines."""
    import numpy as np

    def _cols(events, key):
        vals = sorted({key(e) for e in train})
        idx = {v: i for i, v in enumerate(vals)}
        return len(vals), np.array(
            [[float(idx.get(key(e), -1) == i) for i in range(len(vals))] for e in events]
        )

    if name == "social":
        return _social_test(train, test)
    if name == "whole":
        tr = np.stack([whole_of(e) for e in train])
        ev = np.stack([whole_of(e) for e in test])
        return tr, ev
    if name == "unordered":

        def key(e):
            return tuple(sorted(comp_of(e)))

        k1, a1 = _cols(train, key)
        k2, a2 = _cols(train, lambda e: round(e["end"] - e["start"]))
        return np.hstack([a1, a2]), _apply_key(
            test, train, [key, lambda e: round(e["end"] - e["start"])]
        )
    if name == "position":
        parts = [lambda e: comp_of(e)[0], lambda e: comp_of(e)[1], lambda e: comp_of(e)[2]]
        cols = [_cols(train, p) for p in parts]
        return (np.hstack([a for _, a in cols]), _apply_key(test, train, parts))
    if name == "ordered":
        k1, a1 = _cols(train, lambda e: tuple(comp_of(e)))
        k2, a2 = _cols(train, lambda e: round(e["end"] - e["start"]))
        return np.hstack([a1, a2]), _apply_key(
            test, train, [lambda e: tuple(comp_of(e)), lambda e: round(e["end"] - e["start"])]
        )
    raise ValueError(name)


def _social_test(train, test):
    import numpy as np

    def oh(events, vals):
        idx = {v: i for i, v in enumerate(vals)}
        return np.array([[float(idx.get(v, -1) == i) for i in range(len(vals))] for v in events])

    callers = sorted({e["caller"] for e in train})
    recips = sorted({e["recipient"] for e in train})
    return (
        np.hstack(
            [oh([e["caller"] for e in train], callers), oh([e["recipient"] for e in train], recips)]
        ),
        np.hstack(
            [oh([e["caller"] for e in test], callers), oh([e["recipient"] for e in test], recips)]
        ),
    )


def _apply_key(events, train, key_fns):
    import numpy as np

    cols = []
    for fn in key_fns:
        vals = sorted({fn(e) for e in train})
        idx = {v: i for i, v in enumerate(vals)}
        cols.append(
            np.array([[float(idx.get(fn(e), -1) == i) for i in range(len(vals))] for e in events])
        )
    return np.hstack(cols)


# ------------------------------------------------------------------ bootstrap


def _paired_boot_ba(correct_a, correct_b, y, blocks, b=BOOT_B):
    """Block-clustered paired bootstrap of Delta = L(a) - L(b) where
    L = 1 - balanced accuracy (the preregistered loss).

    ``correct_*`` are per-event 0/1 correctness vectors; balanced accuracy is
    recomputed per replicate from the resampled events (class-aware).
    """
    import numpy as np

    ca, cb = np.asarray(correct_a), np.asarray(correct_b)
    ub = np.unique(blocks)
    blk_idx = {bl: np.where(blocks == bl)[0] for bl in ub}

    def _ba(corr, y_sub):
        return float(np.mean([corr[y_sub == cl].mean() for cl in np.unique(y_sub)]))

    delta_obs = (1 - _ba(ca, y)) - (1 - _ba(cb, y))  # = BA(b) - BA(a)
    rng = np.random.RandomState(SEED)
    reps = []
    for _ in range(b):
        pick = ub[rng.randint(0, len(ub), len(ub))]
        idx = np.concatenate([blk_idx[bl] for bl in pick])
        # replicate size is random under block resampling: mask classes on
        # the RESAMPLED labels, never the full-length y
        reps.append((1 - _ba(ca[idx], y[idx])) - (1 - _ba(cb[idx], y[idx])))
    ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    return {
        "mean_delta_L": float(delta_obs),
        "ci95": ci,
        "sign_note": "positive = the ordered (organisation-sensitive) model "
        "has the lower preregistered loss",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    _real_run()


def _self_test() -> None:
    """Synthetic component system: the suffix component carries the target."""
    import numpy as np

    from bioacoustic_pipeline.ingestion.frozen_dusted import FrozenDUSTEDModel

    rng = np.random.RandomState(0)
    base_pre = [rng.randn(44) * 3.0 for _ in range(2)]
    base_suf = [rng.randn(44) * 3.0 for _ in range(2)]
    events = []
    for k in range(160):
        pre = rng.randint(0, 2)
        suf = rng.randint(0, 2)  # suffix carries the target
        third_vecs = [
            base_pre[pre] + 0.2 * rng.randn(44),
            0.2 * rng.randn(44),
            base_suf[suf] + 0.2 * rng.randn(44),
        ]
        frames = np.vstack([np.tile(v, (4, 1)) for v in third_vecs])
        events.append(
            {
                "file": f"synth{k}.wav",
                "caller": 1 + k % 4,
                "recipient": 10,
                "context": 11 + suf,
                "block": f"b{k % 6}",
                "start": 0,
                "end": 12,
                "frames": frames,
            }
        )
    thirds = np.vstack([_thirds_mean(e["frames"]) for e in events])
    vocab = FrozenDUSTEDModel(n_clusters=4, random_state=42).fit(thirds)
    comp = {id(e): tuple(int(c) for c in vocab.assign(_thirds_mean(e["frames"]))) for e in events}
    y = np.array([e["context"] for e in events])

    def _majority_acc(ids):
        pred = np.zeros_like(y)
        for c in np.unique(ids):
            m = ids == c
            pred[m] = np.bincount(y[m]).argmax()
        return _balanced_acc(y, pred)

    suf_ids = np.array([comp[id(e)][2] for e in events])
    pre_ids = np.array([comp[id(e)][0] for e in events])
    assert _majority_acc(suf_ids) > 0.7, "suffix component not decoded"
    assert _majority_acc(pre_ids) < 0.7, "prefix leaked the target"
    print(
        f"SELF-TEST OK (suffix decodes target acc={_majority_acc(suf_ids):.2f}, "
        f"prefix acc={_majority_acc(pre_ids):.2f})",
        flush=True,
    )


def _real_run() -> None:
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import FEATURE_SCHEMA_VERSION
    from bioacoustic_pipeline.ingestion.frozen_dusted import FrozenDUSTEDModel
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )

    prereg = prereg_spec()
    events = parent_events_from_caches()
    if not events:
        sys.exit("no parent events found")
    # the manifest is per SOURCE FILE: a multi-call source contributes several
    # parent events, all inheriting their file's partition (never both sides)
    file_part = {
        f: "eval"
        if int(hashlib.sha1(blk.encode()).hexdigest()[:8], 16) % 100 >= 100 * TRAIN_FRAC
        else "fit"
        for f, blk in {e["file"]: e["block"] for e in events}.items()
    }
    uniq_files = sorted(file_part)
    parts = [file_part[f] for f in uniq_files]
    blocks_u = [next(e["block"] for e in events if e["file"] == f) for f in uniq_files]
    contract = ScientificRunContract(
        run_name="component_study",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        masks={
            "source_validity": MaskProvenance(
                kind="source_validity",
                produced_by="fingerprint absolute gate",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
            "detected_activity": MaskProvenance(
                kind="detected_activity",
                produced_by="valid-span parent events",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
            "feature_support": MaskProvenance(
                kind="feature_support",
                produced_by="contiguous valid spans",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(uniq_files, parts, blocks_u),
        eval_split={"provenance": "recording-block hash split 70/30", "unit": "recording block"},
    )
    enforcement = raise_for_science(contract)

    train, test = block_split(events)
    print(f"parent events: {len(events)} (train {len(train)} / eval {len(test)})", flush=True)

    # frozen vocabulary on TRAIN thirds only
    thirds_tr = np.vstack([_thirds_mean(e["frames"]) for e in train])
    vocab = FrozenDUSTEDModel(n_clusters=K_VOCAB, random_state=42).fit(thirds_tr)
    comp = {id(e): tuple(int(c) for c in vocab.assign(_thirds_mean(e["frames"]))) for e in events}
    whole = {id(e): e["frames"].mean(axis=0) for e in events}

    # reuse: prefix/suffix class independence across ALL events
    pre = np.array([comp[id(e)][0] for e in events])
    suf = np.array([comp[id(e)][2] for e in events])
    table = np.zeros((K_VOCAB, K_VOCAB))
    for a, b in zip(pre, suf):
        table[a, b] += 1
    obs_chi = float(
        (
            (table - table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / table.sum()) ** 2
            / (table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / table.sum() + 1e-12)
        ).sum()
    )
    rng = np.random.RandomState(SEED)
    nul = []
    flat = table.flatten()
    for _ in range(PERM_B):
        t = rng.permutation(flat).reshape(K_VOCAB, K_VOCAB)
        exp = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / t.sum()
        nul.append(float(((t - exp) ** 2 / (exp + 1e-12)).sum()))
    reuse = {
        "n_combinations_observed": int((table > 0).sum()),
        "possible": K_VOCAB * K_VOCAB,
        "chi_square": round(obs_chi, 2),
        "perm_p_independence": round(
            float((1 + np.sum(np.asarray(nul) >= obs_chi)) / (PERM_B + 1)), 5
        ),
        "note": "independence p HIGH = components recur in different "
        "combinations (reuse); p LOW = positions coupled",
    }

    # supported cells: caller x recipient with both contexts >= MIN_CALLS
    cells: dict = {}
    for e in events:
        cells.setdefault((e["caller"], e["recipient"]), {11: 0, 12: 0})
        cells[(e["caller"], e["recipient"])][e["context"]] += 1
    supported_keys = {
        k for k, v in cells.items() if all(v[c] >= MIN_CALLS_PER_CELL for c in (11, 12))
    }
    tr = [e for e in train if (e["caller"], e["recipient"]) in supported_keys]
    ev = [e for e in test if (e["caller"], e["recipient"]) in supported_keys]
    print(f"supported cells: {len(supported_keys)} | eval events {len(ev)}", flush=True)

    def comp_of(e):
        return comp[id(e)]

    def whole_of(e):
        return whole[id(e)]

    suite = _fit_predict_suite(tr, ev, lambda name, a, b: _design(name, a, b, comp_of, whole_of))

    # per-event outcomes for the paired bootstrap of the PREREGISTERED
    # metric (L = 1 - balanced accuracy): per-event correctness + class,
    # resampled by recording block; balanced accuracy is computed per
    # replicate (raw 0/1 loss would be a different, imbalance-sensitive L)
    import numpy as np

    y_ev = np.array([e["context"] for e in ev])
    classes = sorted(set(y_ev.tolist()))
    correct = {}
    for name in ("unordered", "ordered"):
        xtr, xev = _design(name, tr, ev, comp_of, whole_of)
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
        y_tr_arr = np.array([e["context"] for e in tr])
        w = _ridge_fit(
            (xtr - mu) / sd,
            np.stack([(y_tr_arr == c).astype(float) for c in classes], axis=1),
        )
        scores = _ridge_predict(w, (xev - mu) / sd)
        pred = np.array([classes[i] for i in scores.argmax(1)])
        correct[name] = (y_ev == pred).astype(float)
    blocks_ev = np.array([e["block"] for e in ev])
    d_org = _paired_boot_ba(correct["unordered"], correct["ordered"], y_ev, blocks_ev)

    # combination holdout: withhold (prefix,suffix) pairs from ALL training
    def pair_key(e):
        return (comp[id(e)][0], comp[id(e)][2])

    pairs = sorted({pair_key(e) for e in events})
    held_pairs = {
        p
        for i, p in enumerate(pairs)
        if int(hashlib.sha1(str(p).encode()).hexdigest()[:8], 16) % 4 == 3
    }
    tr_h = [e for e in tr if pair_key(e) not in held_pairs]
    ev_h = [e for e in ev if pair_key(e) in held_pairs]
    combo = {
        "n_pairs_total": len(pairs),
        "n_pairs_held": len(held_pairs),
        "n_eval_events_on_held": len(ev_h),
    }
    if len(ev_h) >= 20 and len(tr_h) >= 40:
        suite_h = _fit_predict_suite(
            tr_h, ev_h, lambda name, a, b: _design(name, a, b, comp_of, whole_of)
        )
        # whole-call template baseline: 1-NN on whole fingerprint
        wtr = np.stack([whole_of(e) for e in tr_h])
        wev = np.stack([whole_of(e) for e in ev_h])
        ytr_h = np.array([e["context"] for e in tr_h])
        nn = np.argmin(((wev[:, None] - wtr[None]) ** 2).mean(axis=2), axis=1)
        nn_pred = ytr_h[nn]
        yev_h = np.array([e["context"] for e in ev_h])
        combo["template_1nn_balanced_acc"] = _balanced_acc(yev_h, nn_pred)
        combo["ordered_balanced_acc_on_held"] = suite_h["ordered"]["balanced_acc"]
        combo["unordered_balanced_acc_on_held"] = suite_h["unordered"]["balanced_acc"]
        combo["position_balanced_acc_on_held"] = suite_h["position"]["balanced_acc"]
    else:
        combo["note"] = "insufficient support for the combination-holdout test"

    ladder = {
        "intra_call_acoustic_organisation": reuse["n_combinations_observed"] >= 8,
        "organisation_associated_information": bool(d_org["ci95"][0] > 0),
        "combination_generalisation_above_templates": bool(
            combo.get("ordered_balanced_acc_on_held", 0) > combo.get("template_1nn_balanced_acc", 1)
        ),
        "receiver_tiers": "require the separate behavioural assay "
        "(RECEIVER_HANDOFF.md) — not tested computationally",
    }
    report = {
        "preregistration": prereg,
        "release_enforcement": enforcement,
        "counts": {
            "parent_events": len(events),
            "train": len(train),
            "eval": len(test),
            "supported_cells": len(supported_keys),
            "eval_events_supported": len(ev),
        },
        "reuse": reuse,
        "models_eval": suite,
        "delta_organisation": d_org,
        "combination_holdout": combo,
        "ladder": ladder,
        "scope_caveats": [
            "parent events are fingerprint-valid spans: the amplitude-"
            "dependent gate scopes the inventory (documented, not corrected)",
            "thirds are a declared modelling choice / baseline, not discovered boundaries",
            "bout-level context labels apply to the bout; the intra-call claim "
            "is about localisable information, not per-segment meaning",
            "predictive evidence only — no words, no sentences, no meaning",
        ],
    }
    out = ROOT / "outputs" / "component_study.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "reuse_p": reuse["perm_p_independence"],
                "models": {k: (v or {}).get("balanced_acc") for k, v in suite.items()},
                "delta_organisation": d_org,
                "combination_holdout": combo,
                "ladder": ladder,
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
