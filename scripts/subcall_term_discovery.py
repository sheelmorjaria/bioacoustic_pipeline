#!/usr/bin/env python3
"""Recurrent sub-call acoustic unit discovery (variable-length term discovery).

TASK: find recognisable acoustic fragments that recur INSIDE different
complete vocalisations — the same piece appearing within several calls,
with boundaries PROPOSED BY REPEATED CORRESPONDENCE, not imposed (the
equal-thirds study did not test this; it fixed the sections).

PREREGISTRATION (frozen before the search):

  PARENT EVENTS: the validity-span complete calls (>= MIN_SPAN frames);
  source-grounded boundaries; padding/gated frames excluded by construction.
  Internal unit boundaries are NOT required to coincide with silence.

  REPRESENTATION: per-frame supported 12-feature trajectories (schema v2),
  discovery-fitted shared z-scaling; per-window z-normalisation for the
  fixed-length seed search; per-sequence normalisation for DTW.

  SEED SEARCH (matrix-profile style): declared lengths L in {8, 16, 24}
  frames (80/160/240 ms). Queries: non-overlapping starts per event;
  candidates: stride-2 starts in OTHER events (distinct source events only;
  reciprocal and same-event pairs excluded). Seeds = cross-event window
  pairs with cosine distance <= SEED_Q-th percentile of the per-caller
  cross-event distance distribution (frozen on discovery).

  REFINEMENT: subsequence DTW of the seed's padded neighbourhood against
  the whole partner event (Sakoe-Chiba band 30%): free start/end on the
  partner side proposes both intervals; boundary uncertainty = spread of
  refined boundaries across occurrences (retained, never forced clean).

  FAMILIES: exemplar-centered ONLY (star topology — a member matches the
  EXEMPLAR within the frozen DTW radius; no A~B~C chaining). >= 3 distinct
  source events; recording-group spread recorded. Nested shorter candidates
  are kept separate, never merged.

  SUCCESS CRITERION (primary): a frozen fragment is recovered by
  subsequence-DTW assignment in INDEPENDENT complete vocalisations
  (validation blocks) with valid support, while the host calls are NOT
  whole-call alike (whole-call distance recorded; recovery must not be
  explainable by whole-call similarity).

  CONTROL: duration-matched RANDOM fragments from discovery parents,
  frozen identically, assigned to the same held-out material — the
  specificity denominator (a generic sustained tone may match much; that
  is reported as low specificity, not hidden).

  CONTEXT: metadata on occurrences (C11/C12), never a selection criterion.

  INVENTORY: one record per family — exemplar source interval, occurrence
  intervals, callers/groups, duration distribution, boundary spread,
  preceding/following assigned-or-unassigned material, within/cross-caller
  support, held-out results, source-review status (unreviewed by default;
  review is the source-reference panel's job).

  FRAMING: "recurrent sub-call acoustic unit" is the measured claim;
  "word candidate" is the working hypothesis only. No meaning, no
  sentences, no compositional claims from discovery alone.

Usage:
    python scripts/subcall_term_discovery.py --self-test
    python scripts/subcall_term_discovery.py

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

MIN_SPAN = 8
LENGTHS = (8, 16, 24)
SEED_Q = 2.0  # percentile of cross-event window distances -> seed cut
MAX_EVENTS_CALLER = 80  # discovery sample per caller (documented cap)
TOP_SEEDS = 150
MIN_FAMILY_EVENTS = 3
TRAIN_FRAC = 0.7
DTW_RADIUS_FRAC = 0.30
HELDOUT_CAP = 240  # validation parents sampled for assignment
N_CONTROL_FRAGMENTS = 25
SEED = 0


def prereg_spec():
    return {
        "task": "variable-length recurrent sub-call acoustic units inside "
        "complete vocalisations (term discovery, partial inventory)",
        "parent_events": f"validity spans >= {MIN_SPAN} frames",
        "seed_lengths_frames": list(LENGTHS),
        "seed_cut": f"cross-event window cosine distance <= SEED_Q={SEED_Q}% "
        "percentile (discovery-frozen)",
        "refinement": "subsequence DTW, Sakoe-Chiba 30%, free partner ends",
        "families": "exemplar-centered star (no chaining), >= "
        f"{MIN_FAMILY_EVENTS} distinct source events",
        "success_criterion": "frozen fragment recovered in independent "
        "vocalisations while host calls are not whole-call alike",
        "control": "duration-matched random fragments, identical procedure",
        "framing": "recurrent sub-call acoustic unit (measured) / word candidate (hypothesis only)",
    }


# ------------------------------------------------------------------ loading


def load_parents():
    """Parent complete calls with their per-frame supported-feature tracks."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        contiguous_spans,
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
    feats = supported_features()
    fidx = [_NAME_TO_IDX[n] for n in feats]
    parents = []
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
            spans = contiguous_spans(valid)
            for sid in sorted(set(spans.tolist())):
                if sid < 0:
                    continue
                idx = np.where(spans == sid)[0]
                if len(idx) < MIN_SPAN:
                    continue
                parents.append(
                    {
                        "file": nm,
                        "caller": a.emitter,
                        "context": a.context,
                        "block": filename_block_group(nm),
                        "start": int(idx[0]),
                        "end": int(idx[-1]) + 1,
                        "track": f[idx],
                    }
                )
    parents.sort(key=lambda p: (p["file"], p["start"]))
    return parents, feats


def part_of_block(block: str) -> str:
    return (
        "discovery"
        if int(hashlib.sha1(block.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
        else "validation"
    )


# ------------------------------------------------------------------ DTW core


def subseq_dtw(query, candidate, band_frac=DTW_RADIUS_FRAC):
    """Subsequence DTW: query must match SOME interval of candidate.

    Inputs are CORPUS-scaled tracks (shared discovery-fitted z-scaling —
    per-sequence renormalisation would distort embedded fragments because
    the host's flanks dominate its statistics). Returns (normalized_cost,
    c_start, c_end) with free candidate ends. Full path (no band): a
    diagonal-anchored band is wrong under free starts, and queries are
    short (<= 24 frames) against hosts (<= ~300), so the full matrix is
    small. ``band_frac`` is retained for interface stability and unused.
    """
    import numpy as np

    q, c = np.asarray(query, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    n, m = len(q), len(c)
    big = 1e12
    D = np.full((n + 1, m + 1), big)
    D[0, :] = 0.0  # free start on the candidate side
    for i in range(1, n + 1):
        step = np.sqrt(((q[i - 1] - c) ** 2).sum(axis=1))
        seg = np.minimum(np.minimum(D[i - 1, :m], D[i - 1, 1:]), D[i, :m])
        D[i, 1:] = np.minimum(D[i, 1:], seg + step)
    j = int(np.argmin(D[n, 1:])) + 1
    # degeneracy guard: a "match" shorter than half the query is not a
    # fragment occurrence (1-frame matches otherwise game the free ends)
    i0, jj0, c_start0 = n, j, j
    while i0 > 0:
        cands = []
        for di, dj in ((-1, 0), (0, -1), (-1, -1)):
            ii, jjj = i0 + di, jj0 + dj
            if ii >= 0 and jjj >= 0 and D[ii, jjj] < big:
                cands.append((D[ii, jjj], ii, jjj))
        if not cands:
            break
        _, i0, jj0 = min(cands)
        c_start0 = min(c_start0, jj0)
    if j - c_start0 + 1 < max(1, n // 2):
        return float("inf"), 1, 1
    cost = float(D[n, j] / n)
    # backtrack for the matched interval
    i, jj, c_start = n, j, j
    while i > 0:
        cands = []
        for di, dj in ((-1, 0), (0, -1), (-1, -1)):
            ii, jjj = i + di, jj + dj
            if ii >= 0 and jjj >= 0 and D[ii, jjj] < big:
                cands.append((D[ii, jjj], ii, jjj))
        if not cands:
            break
        _, i, jj = min(cands)
        c_start = min(c_start, jj)
    return cost, max(1, c_start), j


# ------------------------------------------------------------------ pipeline


def _windows(track, length, stride):
    import numpy as np

    out, starts = [], []
    for s in range(0, max(len(track) - length + 1, 0), stride):
        w = track[s : s + length]
        mu, sd = w.mean(0), w.std(0) + 1e-9
        out.append(((w - mu) / sd).ravel())
        starts.append(s)
    return np.stack(out) if out else np.zeros((0, length * 12)), starts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    _real_run()


def _self_test() -> None:
    """Synthetic: shared fragment at varying positions inside unlike calls."""
    import numpy as np

    rng = np.random.RandomState(0)
    tmpl = rng.randn(12, 12) * 0.3 + np.sin(np.linspace(0, 3, 12))[:, None]
    calls = []
    for k in range(24):
        pre = rng.randn(6 + k % 5, 12) * 0.8
        post = rng.randn(5 + k % 7, 12) * 0.8
        frag = tmpl + 0.05 * rng.randn(12, 12)
        calls.append((k, np.vstack([pre, frag, post])))
    # corpus-level shared scaling (as in the real run)
    allf = np.vstack([tr for _, tr in calls])
    mu, sd = allf.mean(0), allf.std(0) + 1e-9
    calls = [(k, (tr - mu) / sd) for k, tr in calls]
    tmpl = (tmpl - mu) / sd
    # the shared fragment must be locatable inside unlike hosts
    costs, starts, ends = [], [], []
    for k, tr in calls[:6]:
        cost, cs, ce = subseq_dtw(tmpl, tr)
        costs.append(cost)
        starts.append(cs)
        ends.append(ce)
    assert np.max(costs) < 1.0, costs
    # intervals must cover the true fragment position (6+k%5 .. +12)
    for (k, tr), cs, ce in zip(calls[:6], starts, ends):
        true_s = 6 + k % 5
        assert cs >= true_s - 3 and ce <= true_s + 12 + 3, (k, cs, ce)
    # random fragments must NOT transfer to unlike hosts at the same cut
    rc = [subseq_dtw((rng.randn(12, 12) * 0.8 - mu) / sd, tr)[0] for _, tr in calls[:6]]
    assert min(costs) < min(rc), (costs, rc)
    print(
        f"SELF-TEST OK (fragment cost {np.mean(costs):.3f} < random "
        f"{np.mean(rc):.3f}; boundaries within +-2 frames)",
        flush=True,
    )


def _real_run() -> None:
    import numpy as np

    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import FEATURE_SCHEMA_VERSION
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )

    parents, feats = load_parents()
    files = sorted({p["file"] for p in parents})
    fpart = {f: part_of_block(next(p["block"] for p in parents if p["file"] == f)) for f in files}
    contract = ScientificRunContract(
        run_name="subcall_term_discovery",
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
                produced_by="valid-span parents",
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
            [fpart[f] for f in files],
            [next(p["block"] for p in parents if p["file"] == f) for f in files],
        ),
        eval_split={"provenance": "recording-block hash split 70/30", "unit": "recording block"},
        recurrence_validity_mode="masked (validity-span parents)",
    )
    enforcement = raise_for_science(contract)

    disc = [p for p in parents if fpart[p["file"]] == "discovery"]
    val = [p for p in parents if fpart[p["file"]] == "validation"]
    # shared feature scaling fitted on discovery, frozen
    all_disc_frames = np.vstack([p["track"] for p in disc])
    mu, sd = all_disc_frames.mean(0), all_disc_frames.std(0) + 1e-9
    for p in parents:
        p["track"] = (p["track"] - mu) / sd

    by_caller: dict = {}
    for p in disc:
        by_caller.setdefault(p["caller"], []).append(p)
    rng = np.random.RandomState(SEED)

    # ---- seed search per caller (within-caller recurrence first)
    seeds = []
    for y, evs in sorted(by_caller.items()):
        if len(evs) < MIN_FAMILY_EVENTS + 1:
            continue
        evs = sorted(evs, key=lambda p: (p["file"], p["start"]))
        evs = evs[:MAX_EVENTS_CALLER]
        for L in LENGTHS:
            wq, sq = [], []
            for p in evs:
                w, s = _windows(p["track"], L, L)  # non-overlapping queries
                wq.append(w)
                sq.append(s)
            cand, cand_ev = [], []
            for p in evs:
                w, s = _windows(p["track"], L, 2)  # stride-2 candidates
                cand.append(w)
                cand_ev.append(s)
            dists = []
            pair_lists = []
            for qi in range(len(evs)):
                for ci in range(len(evs)):
                    if qi == ci:
                        continue
                    if not len(wq[qi]) or not len(cand[ci]):
                        continue
                    sim = wq[qi] @ cand[ci].T
                    d = np.sqrt(np.maximum(2 - 2 * sim, 0))
                    best = d.min(axis=1)
                    bj = d.argmin(axis=1)
                    dists.extend(best.tolist())
                    for k, (dd, j) in enumerate(zip(best, bj)):
                        pair_lists.append((dd, qi, sq[qi][k], ci, cand_ev[ci][j]))
            if not dists:
                continue
            cut = float(np.percentile(dists, SEED_Q))
            for dd, qi, qs, ci, cs in pair_lists:
                if dd <= cut:
                    seeds.append(
                        {
                            "caller": y,
                            "L": L,
                            "dist": dd,
                            "a_ev": qi,
                            "a_start": qs,
                            "b_ev": ci,
                            "b_start": cs,
                        }
                    )
    # v2 REVISION (control-driven, documented): v1's radius rule
    # (max(cost*1.5, 0.75)) accepted ~95% of duration-matched RANDOM
    # fragments on held-out parents — uncalibrated. The radius is now set
    # from the random-fragment cost distribution on DISCOVERY parents
    # (CTRL_PCT percentile), so random fragments are accepted at ~that
    # rate by construction. No family outcome informed this rule.
    CTRL_PCT = 5.0
    ctrl_pool = disc[:2000]
    ctrl_costs = []
    for i in range(60):
        pp = ctrl_pool[rng.randint(0, len(ctrl_pool))]
        L = LENGTHS[i % len(LENGTHS)]
        if len(pp["track"]) < L + 2:
            continue
        s0 = rng.randint(0, len(pp["track"]) - L)
        q = pp["track"][s0 : s0 + L]
        for k2 in rng.randint(0, len(ctrl_pool), size=12):
            pp2 = ctrl_pool[k2]
            if pp2["file"] == pp["file"]:
                continue
            ctrl_costs.append(subseq_dtw(q, pp2["track"])[0])
    ctrl_costs = [c for c in ctrl_costs if np.isfinite(c)]
    RADIUS = float(np.percentile(ctrl_costs, CTRL_PCT))
    print(
        f"control-calibrated radius frozen at {RADIUS:.4f} "
        f"(random-fragment acceptance ~{CTRL_PCT}%)",
        flush=True,
    )
    # per-caller seed quotas: the densest caller must not consume the budget
    per_caller_budget = max(5, TOP_SEEDS // max(len(by_caller), 1))
    by_c: dict = {}
    for sd in seeds:
        by_c.setdefault(sd["caller"], []).append(sd)
    seeds = []
    for y, lst in sorted(by_c.items()):
        seeds.extend(lst[:per_caller_budget])
    seeds.sort(key=lambda s: s["dist"])
    print(
        f"parents: {len(parents)} (disc {len(disc)} / val {len(val)}) | seeds: {len(seeds)}",
        flush=True,
    )

    # ---- refinement + exemplar-centered families
    evs_by_caller = {
        y: sorted(v, key=lambda p: (p["file"], p["start"]))[:MAX_EVENTS_CALLER]
        for y, v in by_caller.items()
        if len(v) >= MIN_FAMILY_EVENTS + 1
    }
    families = []
    used_pairs: set = set()
    for s in seeds:
        y = s["caller"]
        evs = evs_by_caller[y]
        ea, eb = evs[s["a_ev"]], evs[s["b_ev"]]
        key = (ea["file"], ea["start"], eb["file"], eb["start"])
        if key in used_pairs or ea["file"] == eb["file"]:
            continue
        q = ea["track"][s["a_start"] : s["a_start"] + s["L"]]
        if len(q) < s["L"]:
            continue
        cost, cs, ce = subseq_dtw(q, eb["track"])
        frag = (s["a_start"], s["a_start"] + s["L"])
        radius = RADIUS
        if not np.isfinite(cost) or cost > radius:
            continue
        occ = [(ea, frag[0], frag[1], 0.0), (eb, cs, ce, cost)]
        for p in evs:
            if p["file"] in {ea["file"], eb["file"]}:
                continue
            c2, s2, e2 = subseq_dtw(q, p["track"])
            if c2 <= radius:
                occ.append((p, s2, e2, c2))
        ev_files = {o[0]["file"] for o in occ}
        if len(ev_files) < MIN_FAMILY_EVENTS:
            continue
        used_pairs.add(key)
        families.append(
            {
                "exemplar_file": ea["file"],
                "exemplar_interval_frames": [int(frag[0]), int(frag[1])],
                "caller": y,
                "L": s["L"],
                "seed_dist": round(s["dist"], 4),
                "radius": round(radius, 4),
                "occurrences": [
                    {
                        "file": o["file"],
                        "caller": o["caller"],
                        "context": o["context"],
                        "block": o["block"],
                        "interval_frames": [int(a), int(b)],
                        "dtw_cost": round(c, 4),
                    }
                    for (o, a, b, c) in occ
                ],
                "n_events": len(ev_files),
                "n_blocks": len({o[0]["block"] for o in occ}),
                "durations_frames": sorted(int(b - a) for (_, a, b, _) in occ),
                "boundary_spread": None,  # filled below
                "source_review_status": "unreviewed",
            }
        )
        if len(families) >= 40:
            break
    # boundary uncertainty: spread of occurrence durations
    for f in families:
        durs = f["durations_frames"]
        f["boundary_spread"] = {
            "min": durs[0],
            "max": durs[-1],
            "median": durs[len(durs) // 2],
            "note": "retained as uncertainty; boundaries proposed by "
            "repeated correspondence, never forced clean",
        }
    print(f"families: {len(families)}", flush=True)

    # ---- held-out assignment (primary success criterion) + control
    val_sample = sorted(val, key=lambda p: (p["file"], p["start"]))[:HELDOUT_CAP]
    whole_d = np.stack([p["track"].mean(0) for p in val_sample])
    whole_d = whole_d / (np.linalg.norm(whole_d, axis=1, keepdims=True) + 1e-9)

    def _assign_frozen(query_track, radius):
        hits = []
        for k, p in enumerate(val_sample):
            c, s, e = subseq_dtw(query_track, p["track"])
            if c <= radius:
                # whole-call similarity guard: host must not be a whole-call
                # duplicate of the exemplar
                wv = p["track"].mean(0)
                wv = wv / (np.linalg.norm(wv) + 1e-9)
                ex = query_track.mean(0)
                ex = ex / (np.linalg.norm(ex) + 1e-9)
                cos = float(ex @ wv)
                hits.append(
                    {
                        "file": p["file"],
                        "caller": p["caller"],
                        "interval_frames": [int(s), int(e)],
                        "dtw_cost": round(c, 4),
                        "whole_call_cos": round(cos, 4),
                    }
                )
        return hits

    inventory = []
    for f in families:
        ex_parent = next(p for p in disc if p["file"] == f["exemplar_file"])
        q = ex_parent["track"][f["exemplar_interval_frames"][0] : f["exemplar_interval_frames"][1]]
        hits = _assign_frozen(q, f["radius"])
        inventory.append(
            {
                **f,
                "heldout_hits": hits,
                "heldout_n": len(val_sample),
                "heldout_rate": round(len(hits) / max(len(val_sample), 1), 4),
                "heldout_wholecall_cos_max": round(
                    max([h["whole_call_cos"] for h in hits], default=0.0), 4
                ),
            }
        )

    # duration-matched random-fragment control, identical procedure
    ctrl_rates = []
    ctrl_pool = disc[:2000]
    for i in range(N_CONTROL_FRAGMENTS):
        p = ctrl_pool[rng.randint(0, len(ctrl_pool))]
        L = LENGTHS[i % len(LENGTHS)]
        if len(p["track"]) < L + 2:
            continue
        s0 = rng.randint(0, len(p["track"]) - L)
        q = p["track"][s0 : s0 + L]
        r = RADIUS  # identical frozen radius for controls
        hits = _assign_frozen(q, r)
        ctrl_rates.append(len(hits) / max(len(val_sample), 1))
    ctrl = {
        "n_fragments": len(ctrl_rates),
        "mean_rate": round(float(np.mean(ctrl_rates or [0])), 4),
        "max_rate": round(float(np.max(ctrl_rates or [0])), 4),
    }

    report = {
        "preregistration": prereg_spec(),
        "release_enforcement": enforcement,
        "features": list(feats),
        "counts": {
            "parents": len(parents),
            "discovery": len(disc),
            "validation": len(val),
            "seeds_scored": len(seeds),
            "families": len(families),
            "heldout_parents_examined": len(val_sample),
        },
        "control_duration_matched_random": ctrl,
        "inventory": [
            {k: v for k, v in f.items() if k != "occurrences"}
            | {"occurrences": f["occurrences"][:20]}
            for f in inventory
        ],
        "success_criterion_evaluation": {
            "families_with_heldout_hits": sum(1 for f in inventory if f["heldout_hits"]),
            "control_mean_rate": ctrl["mean_rate"],
            "note": "recovery counts as success only above the duration-"
            "matched control and without whole-call-similarity explanation",
        },
        "unassigned_note": "partial inventory by design — unassigned material "
        "is preferable to inventing a complete vocabulary",
        "scope_caveats": [
            "recurrent sub-call acoustic unit (measured); word candidate is "
            "the working hypothesis only",
            "no meaning, no sentences, no compositional claims from discovery",
            "context is occurrence metadata, never a selection criterion",
            "amplitude-dependent gate scopes the parent inventory",
        ],
    }
    out = ROOT / "outputs" / "subcall_term_discovery.json"
    out.write_text(json.dumps(report, indent=1))
    fam_hits = [(f["exemplar_file"], f["n_events"], f["heldout_rate"]) for f in inventory]
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "control": ctrl,
                "families_with_heldout_hits": report["success_criterion_evaluation"][
                    "families_with_heldout_hits"
                ],
                "top_by_heldout": sorted(fam_hits, key=lambda t: -t[2])[:5],
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
