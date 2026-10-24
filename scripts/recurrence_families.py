#!/usr/bin/env python3
"""Caller-by-context recurrence families (bounded first run, three-way design).

PREREGISTRATION (frozen before any search):

  QUESTIONS (kept separate):
    Q1 within-caller recurrence — does caller y repeat a vocal form in C11?
    Q2 cross-caller correspondence — do other callers produce similar forms
       in C11 (sharing, coexisting with caller-specific frequency)?
    Q3 context association — is a family's use greater in C11 than C12
       WITHIN the same callers (beyond caller composition)?

  ELIGIBLE EVENT (validity contract): a contiguous fingerprint-valid span
  >= MIN_SPAN frames (complete calls, not sliding windows); source valid,
  activity-supported and representation-supported by construction. The
  rejected one-valid-frame windows remain negative audit examples, never
  seeds. Context labels are BOUT-LEVEL: claims read "calls occurring in
  C-labelled bouts".

  REPRESENTATION: one shared whole-call fingerprint (mean over the span's
  valid frames), z-scaled with statistics fitted on DISCOVERY blocks ONLY
  and reused unchanged everywhere (no per-caller coordinates).

  FAMILY DISCOVERY (C11, discovery partition only): radius-to-EXEMPLAR
  grouping around actual source events — a member is within tau of the
  EXEMPLAR, never chained transitively (the DUSTED connected-component
  lesson). tau is frozen as the TAU_Q-th percentile of within-caller
  nearest-neighbour distances on discovery. Two routes, both reported:
    (a) within-caller: per caller with >= MIN_CALLS_EVENTS events;
    (b) caller-balanced pooled: shared-form candidates spanning >= 2 callers.

  ASSIGNMENT (frozen): an eligible call is an occurrence of family m iff
  its scaled fingerprint lies within tau of m's exemplar. No refitting, no
  family editing to improve separation.

  TRANSFER (independent validation blocks): within-caller transfer rate
  (family of y matching y's new recordings) and cross-caller transfer
  (families discovered WITHOUT caller z matching z's new recordings), each
  with explicit denominators. Search-mode comparisons use MATCHED gallery
  sizes (nearest-neighbour minima shrink with gallery size — never compared
  without exposure).

  CONTEXT ASSOCIATION: r_myc = eligible bouts (files) containing >= 1
  accepted occurrence / eligible bouts examined, per family x caller x
  context; Delta_my = r_my,C11 - r_my,C12. Null: context labels permuted at
  the FILE level within caller (preserves caller and file composition).
  Bout-level rates prevent burst domination; call-level rates reported
  separately.

  EVIDENCE RECORD per family: recurrence support (calls, bouts, blocks),
  caller reach, opportunity-adjusted usage, cross-caller transfer, context
  association, source-review status ("unreviewed" — review is the
  source-reference panel's job, never automatic).

  FRAMING: acoustic recurrence and context association under a declared
  procedure — NOT meaning, NOT 'distinctive signal' claims; absence of
  cross-caller matches is reported as "no correspondence detected under the
  tested representation, threshold and sampling".

Usage:
    python scripts/recurrence_families.py --self-test
    python scripts/recurrence_families.py

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

MIN_SPAN = 6
MIN_EVENTS_CALLER = 12  # within-caller discovery needs repeated opportunity
MIN_FAMILY = 3  # pooled families need >= 3 members
TAU_Q = 25.0  # percentile of within-caller NN distances -> tau
TRAIN_FRAC = 0.7
PERM_B = 2000
SEED = 0
PRIMARY_CTX = 11
OTHER_CTX = 12
MIN_FILES_PER_CTX = 5  # caller enters the association test with this support
MIN_CALLERS_ASSOC = 3


def prereg_spec(tau_note="set on discovery (TAU_Q percentile of within-caller NN distances)"):
    return {
        "design": "three-way: within-caller recurrence / cross-caller "
        "correspondence in the same context / context association within "
        "the same callers",
        "eligible_event": f"fingerprint-valid span >= {MIN_SPAN} frames "
        "(complete call; bout-level context scope)",
        "representation": "shared whole-call fingerprint, discovery-fitted "
        "z-scaling frozen for all later use",
        "family_rule": f"radius-to-exemplar <= tau (tau {tau_note}); NO transitive chaining",
        "discovery_routes": [
            f"within-caller (per caller >= {MIN_EVENTS_CALLER} C11 events)",
            f"caller-balanced pooled (>= {MIN_FAMILY} members, >= 2 callers)",
        ],
        "assignment": "frozen: distance-to-exemplar <= tau",
        "transfer": "independent validation blocks; matched gallery sizes "
        "for search-mode comparisons; denominators always reported",
        "context_association": "r_myc bout-level; Delta_my within caller; "
        "file-level context permutation within caller (caller-preserving)",
        "framing": "acoustic recurrence + context association under a "
        "declared procedure — not meaning",
    }


def load_events() -> list[dict]:
    """Eligible complete calls from both frozen caches (deterministic order)."""
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
            if a.context not in (PRIMARY_CTX, OTHER_CTX) or a.addressee <= 0:
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
                vec = f[idx].mean(axis=0)
                if not np.isfinite(vec).all():
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
                        "vec": vec,
                    }
                )
    events.sort(key=lambda e: (e["file"], e["start"]))
    return events


def block_of(block: str) -> str:
    return (
        "discovery"
        if int(hashlib.sha1(block.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
        else "validation"
    )


def _radius_families(items, tau, min_size, min_callers=1):
    """Radius-to-exemplar grouping, densest-first seeding (rule v2).

    A member is within tau of the EXEMPLAR only — never chained transitively
    (the DUSTED connected-component lesson). v1 seeded in file order and
    produced zero families at the frozen tau (the dense modes were never
    seeded); revised to densest-first BEFORE any endpoint was evaluated —
    no thresholds, rates or outcomes were consulted.
    """
    import numpy as np

    n = len(items)
    if n < min_size:
        return []
    m = np.stack([it["z"] if "z" in it else it["vec"] for it in items])
    # blocked distance computation -> boolean neighbour matrix
    nb = np.zeros((n, n), dtype=bool)
    for s in range(0, n, 1024):
        e = min(s + 1024, n)
        d = np.linalg.norm(m[s:e, None] - m[None], axis=2)
        nb[s:e] = d <= tau
    np.fill_diagonal(nb, True)
    alive = np.ones(n, dtype=bool)
    fams = []
    while True:
        degree = nb[alive][:, alive].sum(axis=1)
        if len(degree) < min_size or degree.max() < min_size:
            break
        idx_alive = np.where(alive)[0]
        seed_local = int(np.argmax(degree))
        seed = int(idx_alive[seed_local])
        member_mask = nb[seed] & alive
        members = [items[i] for i in np.where(member_mask)[0]]
        if len(members) >= min_size and len({x["caller"] for x in members}) >= min_callers:
            fams.append(
                {
                    "exemplar": items[seed],
                    "members": members,
                    "size": len(members),
                    "callers": sorted({x["caller"] for x in members}),
                    "blocks": sorted({x["block"] for x in members}),
                    "files": sorted({x["file"] for x in members}),
                }
            )
        alive &= ~member_mask
        if alive.sum() < min_size:
            break
    return fams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    _real_run()


def _self_test() -> None:
    """Synthetic: a shared form in 3 callers in C11 only, plus noise."""
    import numpy as np

    rng = np.random.RandomState(0)
    shared = rng.randn(44) * 0.5 + 10.0
    events = []
    for caller in (1, 2, 3, 4):
        for ctx in (11, 12):
            for k in range(12):
                if caller in (1, 2, 3) and ctx == 11 and k < 5:
                    vec = shared + 0.05 * rng.randn(44)  # the shared C11 form
                else:
                    vec = rng.randn(44) * 2.0
                events.append(
                    {
                        "file": f"f{caller}_{ctx}_{k}",
                        "caller": caller,
                        "recipient": 9,
                        "context": ctx,
                        "block": f"b{caller}_{k % 3}",
                        "start": 0,
                        "end": 8,
                        "vec": vec,
                    }
                )
    tau = 0.8
    c11 = [e for e in events if e["context"] == 11]
    fams = _radius_families(c11, tau, min_size=3, min_callers=2)
    assert fams, "shared form not discovered"
    big = max(fams, key=lambda f: f["size"])
    assert set(big["callers"]) == {1, 2, 3}, big["callers"]
    assert big["size"] >= 12
    print(f"SELF-TEST OK (shared form: {big['size']} events, callers {big['callers']})", flush=True)


def _real_run() -> None:
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import FEATURE_SCHEMA_VERSION
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )

    events = load_events()
    files = sorted({e["file"] for e in events})
    file_part = {f: block_of(next(e["block"] for e in events if e["file"] == f)) for f in files}
    contract = ScientificRunContract(
        run_name="recurrence_families",
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
                produced_by="valid-span events",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
            "feature_support": MaskProvenance(
                kind="feature_support",
                produced_by=f"contiguous valid spans >= {MIN_SPAN} frames",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(
            files,
            [file_part[f] for f in files],
            [next(e["block"] for e in events if e["file"] == f) for f in files],
        ),
        eval_split={
            "provenance": "recording-block hash split "
            f"{int(TRAIN_FRAC * 100)}/{int((1 - TRAIN_FRAC) * 100)} "
            "(discovery/validation)",
            "unit": "recording block",
        },
        recurrence_validity_mode="masked (validity-span events; the legacy "
        "unmasked path is refused by contract)",
    )
    enforcement = raise_for_science(contract)

    disc = [e for e in events if file_part[e["file"]] == "discovery"]
    val = [e for e in events if file_part[e["file"]] == "validation"]
    print(
        f"eligible events: {len(events)} (discovery {len(disc)} / validation {len(val)})",
        flush=True,
    )

    # shared scaling, fitted on DISCOVERY only, frozen for everything later
    mu, sd = (
        np.stack([e["vec"] for e in disc]).mean(0),
        (np.stack([e["vec"] for e in disc]).std(0) + 1e-9),
    )
    for e in events:
        e["z"] = (e["vec"] - mu) / sd

    # ---- tau frozen from discovery within-caller NN distances (C11)
    c11_disc = [e for e in disc if e["context"] == PRIMARY_CTX]
    by_caller: dict = {}
    for e in c11_disc:
        by_caller.setdefault(e["caller"], []).append(e)
    nn_d = []
    for y, es in by_caller.items():
        if len(es) < 3:
            continue
        m = np.stack([e["z"] for e in es])
        d = np.linalg.norm(m[:, None] - m[None], axis=2) + np.eye(len(m)) * 1e9
        nn_d.extend(d.min(axis=1).tolist())
    tau = float(np.percentile(nn_d, TAU_Q))
    print(
        f"tau frozen at {tau:.4f} (TAU_Q={TAU_Q} of {len(nn_d)} within-caller NN distances)",
        flush=True,
    )

    # ---- discovery route A: within-caller families
    fams_within = []
    for y, es in sorted(by_caller.items()):
        if len(es) < MIN_EVENTS_CALLER:
            continue
        es = sorted(es, key=lambda e: (e["file"], e["start"]))
        for fam in _radius_families(es, tau, min_size=3, min_callers=1):
            fam["route"] = f"within-caller y={y}"
            fams_within.append(fam)

    # ---- discovery route B: caller-balanced pooled families
    pooled_sorted = sorted(c11_disc, key=lambda e: (e["file"], e["start"]))
    fams_pooled = _radius_families(pooled_sorted, tau, min_size=MIN_FAMILY, min_callers=2)
    for fam in fams_pooled:
        fam["route"] = "caller-balanced pooled"

    print(
        f"discovery: {len(fams_within)} within-caller families, "
        f"{len(fams_pooled)} pooled shared-form families",
        flush=True,
    )

    # ---- frozen assignment function
    def assign(ev, fam):
        return float(np.linalg.norm(ev["z"] - fam["exemplar"]["z"])) <= tau

    # ---- transfer on validation blocks (denominators explicit)
    c11_val = [e for e in val if e["context"] == PRIMARY_CTX]
    transfer = {"within_caller": [], "cross_caller": []}
    for fam in fams_within:
        y = fam["exemplar"]["caller"]
        elig = [e for e in c11_val if e["caller"] == y]
        if not elig:
            continue
        hits = sum(1 for e in elig if assign(e, fam))
        # DIAGNOSTIC (descriptive, non-frozen rule): nearest-MEMBER
        # acceptance vs nearest-EXEMPLAR. d_NN <= d_family whenever the
        # exemplar is a subset of the members, so the two statistics can
        # legitimately diverge; this documents which one the frozen
        # transfer number measures (nearest-exemplar).
        mz = np.stack([m["z"] for m in fam["members"]])
        member_hits = sum(
            1 for e in elig if float(np.linalg.norm(mz - e["z"], axis=1).min()) <= tau
        )
        transfer["within_caller"].append(
            {
                "family": fam["route"] + " @ " + fam["exemplar"]["file"],
                "caller": y,
                "hits": hits,
                "eligible": len(elig),
                "rate": round(hits / len(elig), 4),
                "nearest_member_rate_descriptive": round(member_hits / len(elig), 4),
            }
        )
    seen_fam_caller = set()
    for fam in fams_pooled:
        for z in sorted({e["caller"] for e in c11_val} - set(fam["callers"])):
            elig = [e for e in c11_val if e["caller"] == z]
            hits = sum(1 for e in elig if assign(e, fam))
            key = (fam["exemplar"]["file"], z)
            if key in seen_fam_caller:
                continue
            seen_fam_caller.add(key)
            transfer["cross_caller"].append(
                {
                    "family": fam["exemplar"]["file"],
                    "new_caller": z,
                    "hits": hits,
                    "eligible": len(elig),
                    "rate": round(hits / len(elig), 4),
                }
            )

    # matched-gallery comparison for search modes (Q1 vs Q2 rates at tau).
    # PAIRED structure: the SAME queries and the SAME random gallery draw
    # feed both conditions; per-query indicators are kept for a paired
    # bootstrap clustered by caller (review correction: the 0.112
    # difference carries its own uncertainty record, separate from any
    # prespecified confirmation).
    rng = np.random.RandomState(SEED)
    mode_rates = {}
    paired_indicators: dict = {}
    for y, es in sorted(by_caller.items()):
        if len(es) < 6:
            continue
        m = np.stack([e["z"] for e in es])
        d_within = (np.linalg.norm(m[:, None] - m[None], axis=2) + np.eye(len(m)) * 1e9).min(axis=1)
        others = [e for e in c11_disc if e["caller"] != y]
        if len(others) < len(es):
            continue
        pick = rng.choice(len(others), size=len(es), replace=False)
        mo = np.stack([others[i]["z"] for i in pick])
        d_cross_matched = np.linalg.norm(m[:, None] - mo[None], axis=2).min(axis=1)
        mode_rates[y] = {
            "within_match_rate": round(float((d_within <= tau).mean()), 4),
            "cross_matched_gallery_rate": round(float((d_cross_matched <= tau).mean()), 4),
            "gallery": len(es),
        }
        paired_indicators[y] = (
            (d_within <= tau).astype(float),
            (d_cross_matched <= tau).astype(float),
        )
    # paired bootstrap over callers (cluster unit), shared draws
    callers_p = sorted(paired_indicators)
    brng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(2000):
        pick = brng.choice(len(callers_p), size=len(callers_p), replace=True)
        w = np.concatenate([paired_indicators[callers_p[i]][0] for i in pick])
        c = np.concatenate([paired_indicators[callers_p[i]][1] for i in pick])
        diffs.append(float(w.mean() - c.mean()))
    obs_diff = float(
        np.mean(
            [paired_indicators[y][0].mean() - paired_indicators[y][1].mean() for y in callers_p]
        )
    )
    matched_gallery_ci = {
        "difference": round(obs_diff, 4),
        "ci95_caller_clustered": [
            round(float(np.percentile(diffs, 2.5)), 4),
            round(float(np.percentile(diffs, 97.5)), 4),
        ],
        "scope": "caller-ASSOCIATED matching advantage at the specified "
        "threshold, equal gallery size, discovery-partition queries; NOT a "
        "confirmed caller-specificity result (no independent split, "
        "discovery-derived threshold, context/recording composition not "
        "matched) — kept separate from prespecified confirmations",
    }

    # ---- context association: r_myc at BOUT (file) level, frozen assignment
    # (families enter in discovery-size order, capped for compute; the cap is
    # part of the declared procedure)
    assoc_report = []
    all_fams = sorted(fams_within + fams_pooled, key=lambda f: -f["size"])[:40]
    # CIRCULARITY CONTROL: families were built from C11 DISCOVERY events, so
    # their own members guarantee C11 hits. The association endpoint uses
    # VALIDATION-partition files only (hits and denominators) — independent
    # of everything the families were constructed from. Discovery-side rates
    # are descriptive membership, never evidence.
    val_events = [e for e in events if file_part[e["file"]] == "validation"]
    file_caller, file_ctx = {}, {}
    for e in val_events:
        file_caller.setdefault(e["file"], e["caller"])
        file_ctx.setdefault(e["file"], e["context"])
    files_by_caller: dict = {}
    for f_, y in file_caller.items():
        files_by_caller.setdefault(y, []).append(f_)
    elig_counts = {  # denominators are fixed under within-caller shuffles
        (y, c): sum(1 for f_ in fs if file_ctx[f_] == c)
        for y, fs in files_by_caller.items()
        for c in (PRIMARY_CTX, OTHER_CTX)
    }
    for fam in all_fams:
        hit_files = set()
        for e in val_events:
            if assign(e, fam):
                hit_files.add(e["file"])
        rates: dict = {}
        for y, fs in files_by_caller.items():
            for c in (PRIMARY_CTX, OTHER_CTX):
                elig = [f_ for f_ in fs if file_ctx[f_] == c]
                if elig:
                    rates[(y, c)] = sum(1 for f_ in elig if f_ in hit_files) / len(elig)
        callers_ok = [
            y
            for (y, c) in rates
            if c == PRIMARY_CTX
            and (y, OTHER_CTX) in rates
            and elig_counts.get((y, PRIMARY_CTX), 0) >= MIN_FILES_PER_CTX
            and elig_counts.get((y, OTHER_CTX), 0) >= MIN_FILES_PER_CTX
        ]
        callers_ok = sorted(set(callers_ok))
        if len(callers_ok) < MIN_CALLERS_ASSOC:
            continue
        obs = float(np.mean([rates[(y, PRIMARY_CTX)] - rates[(y, OTHER_CTX)] for y in callers_ok]))
        # null: file-level context shuffle WITHIN caller (caller/file
        # composition preserved; denominators therefore fixed)
        prng = np.random.RandomState(SEED)
        per_caller = {
            y: (
                np.array([1.0 if f_ in hit_files else 0.0 for f_ in fs]),
                np.array([1 if file_ctx[f_] == PRIMARY_CTX else 0 for f_ in fs]),
            )
            for y, fs in ((y, files_by_caller[y]) for y in callers_ok)
        }
        n_ge = 0
        for _ in range(PERM_B):
            deltas_null = []
            for y, (h, is_p) in per_caller.items():
                lbl = prng.permutation(is_p)
                n11 = max(int(lbl.sum()), 1)
                n12 = max(int((1 - lbl).sum()), 1)
                deltas_null.append(h[lbl == 1].sum() / n11 - h[lbl == 0].sum() / n12)
            if float(np.mean(deltas_null)) >= obs:
                n_ge += 1
        p = (1 + n_ge) / (PERM_B + 1)
        # assessability: with transfer this low, "no association detected"
        # must be separated from "insufficient accepted occurrences to
        # assess" — unavailable tests stay unavailable, never negative
        n_accepted_val = sum(1 for e in val_events if assign(e, fam))
        assoc_report.append(
            {
                "family": fam["exemplar"]["file"],
                "route": fam["route"],
                "size": fam["size"],
                "callers": fam["callers"],
                "n_accepted_validation_occurrences": int(n_accepted_val),
                "assessable": bool(n_accepted_val >= 5),
                "delta_callers": {
                    str(y): round(rates[(y, PRIMARY_CTX)] - rates[(y, OTHER_CTX)], 4)
                    for y in callers_ok
                },
                "n_callers_assoc": len(callers_ok),
                "mean_delta": round(obs, 4),
                "perm_p_file_level_within_caller": round(p, 5),
                "source_review_status": "unreviewed",
            }
        )
    assoc_report.sort(key=lambda r: -r["mean_delta"])

    report = {
        "preregistration": prereg_spec(
            f"frozen at {tau:.4f} = TAU_Q {TAU_Q}% of discovery within-caller NN distances"
        ),
        "release_enforcement": enforcement,
        "counts": {
            "eligible_events": len(events),
            "discovery": len(disc),
            "validation": len(val),
            "c11_discovery_events": len(c11_disc),
            "callers_in_c11_discovery": len(by_caller),
            "within_caller_families": len(fams_within),
            "pooled_families": len(fams_pooled),
        },
        "tau": round(tau, 4),
        "families": [
            {
                "exemplar_file": f["exemplar"]["file"],
                "exemplar_interval_frames": [f["exemplar"]["start"], f["exemplar"]["end"]],
                "route": f["route"],
                "size": f["size"],
                "callers": f["callers"],
                "n_files": len(f["files"]),
                "n_blocks": len(f["blocks"]),
                "source_review_status": "unreviewed",
            }
            for f in (fams_within + fams_pooled)
        ],
        "transfer": {
            "within_caller": transfer["within_caller"][:40],
            "cross_caller": transfer["cross_caller"][:40],
            "n_within_records": len(transfer["within_caller"]),
            "n_cross_records": len(transfer["cross_caller"]),
        },
        "matched_gallery_search_modes": mode_rates,
        "matched_gallery_paired": matched_gallery_ci,
        "assignment_rule_note": "the frozen transfer statistic measures "
        "nearest-EXEMPLAR assignment (distance to the family center <= tau); "
        "nearest-member rates are reported as a descriptive diagnostic",
        "context_association": assoc_report[:40],
        "negative_result_label": "no cross-caller correspondence detected "
        "under the tested representation, threshold and sampling",
        "scope_caveats": [
            "bout-level context labels: claims are 'calls occurring in C-labelled bouts'",
            "shared form and caller-specific frequency can coexist and are reported separately",
            "the amplitude-dependent gate scopes the eligible inventory",
            "recurrence/context association under a declared procedure — "
            "not meaning, not 'distinctive signal' claims",
        ],
    }
    out = ROOT / "outputs" / "recurrence_families.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "tau": report["tau"],
                "transfer_within_mean_rate": round(
                    float(np.mean([t["rate"] for t in transfer["within_caller"]] or [0])), 4
                ),
                "transfer_cross_mean_rate": round(
                    float(np.mean([t["rate"] for t in transfer["cross_caller"]] or [0])), 4
                ),
                "n_assoc_families": len(assoc_report),
                "top_assoc": assoc_report[:3],
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
