#!/usr/bin/env python3
"""Known-fragment sensitivity and source-correspondence benchmark.

PURPOSE (review-directed): before any new discovery version, establish
whether the FROZEN machinery can (1) locate known recurring fragments
within different surrounding material, (2) keep occurrences of the same
fragment together without absorbing acoustically different fragments, and
(3) recover their boundaries and independent occurrences — the matching /
grouping / segmentation separation used in spoken-term-discovery
evaluation, on our own material with constructed ground truth.

DESIGN (frozen before running):
  MATERIAL: caller 215's tracks (the individual-conditioned study's
  caller). KNOWN FRAGMENTS: N_FRAGS random windows (lengths 8/16/24) from
  DISCOVERY-stage tracks — ground-truth identity + source interval.
  HOSTS: real VALIDATION-stage tracks; a random interval of matching
  length is REPLACED by the fragment (track-level splice, 2-frame linear
  crossfade to avoid discontinuity artifacts — the same treatment for
  every condition). Ground-truth interval known exactly.

  MACHINERY UNDER TEST, unchanged: subsequence DTW (free ends, degeneracy
  guard) at the caller-215 study's frozen radius (1.0796) and the frozen
  common corpus scaling.

  MEASURES:
    M1 localization — for accepted hits of the TRUE fragment in its host:
       start/end boundary error (frames) vs ground truth.
    M2 grouping — acceptance of the TRUE fragment vs acceptance of DECOY
       fragments (the other known fragments) on the same implanted hosts,
       plus argmin-identity: which fragment does the host best match?
    M3 occurrence recovery — fraction of implanted hosts where the true
       fragment is accepted.
    M4 unmodified-host reference — fragment acceptance on UNMODIFIED
       validation tracks (ordinary matchability, no implant).

  TRIAGE (review): good localization + good grouping => the machinery can
  find known fragments, and the enrichment CRITERION is the likely
  limitation; decoys absorbed equally => representation/matching
  limitation; known positives not recovered => sensitivity failure.

  This benchmark has constructed ground truth by design; it validates the
  SEARCH, not the existence of biological units.

Usage:
    python scripts/fragment_sensitivity_benchmark.py --self-test
    python scripts/fragment_sensitivity_benchmark.py

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

N_FRAGS = 20
N_HOSTS_EACH = 12
FROZEN_RADIUS = 1.0796
LENGTHS = (8, 16, 24)
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    _real_run()


def _self_test() -> None:
    """Splice/retrieve mechanics on synthetic tracks."""
    import numpy as np

    import subcall_term_discovery as std

    rng = np.random.RandomState(0)
    frag = rng.randn(12, 12) * 0.2 + np.sin(np.linspace(0, 3, 12))[:, None]
    hosts = []
    for k in range(6):
        h = rng.randn(30, 12) * 0.8
        pos = 5 + 2 * k
        h[pos : pos + 12] = frag
        hosts.append((h, pos))
    for h, pos in hosts:
        cost, s, e = std.subseq_dtw(frag, h)
        assert cost < 1.0 and abs(s - pos) <= 2 and abs(e - (pos + 12)) <= 2, (cost, s, e, pos)
    print("SELF-TEST OK (synthetic splice located within +-2 frames)", flush=True)


def _real_run() -> None:
    import numpy as np

    import individual_repertoire as ir
    import subcall_term_discovery as std
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
    )

    parents, feats = std.load_parents()
    mine = [p for p in parents if p["caller"] == 215]
    if not mine:
        sys.exit("caller 215 material unavailable")
    for p in mine:
        p["stage"] = ir._stage(p["block"])
    disc = [p for p in mine if p["stage"] == "discovery"]
    val = [p for p in mine if p["stage"] == "validation"]
    disc = sorted(disc, key=lambda p: (p["file"], p["start"]))
    val = sorted(val, key=lambda p: (p["file"], p["start"]))
    # frozen common corpus scaling (as all prior studies)
    pooled = [p for p in parents if std.part_of_block(p["block"]) == "discovery"]
    allf = np.vstack([p["track"] for p in pooled])
    mu, sd = allf.mean(0), allf.std(0) + 1e-9
    for p in mine:
        p["z"] = (p["track"] - mu) / sd

    contract = ScientificRunContract(
        run_name="fragment_sensitivity_benchmark",
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
        },
        source_manifest={p["file"]: {"group": p["block"], "part": "benchmark"} for p in mine[:1]},
        eval_split={
            "provenance": "constructed ground-truth benchmark; no fitting, frozen radius reused",
            "unit": "n/a",
        },
    )
    enforcement = raise_for_science(contract)

    rng = np.random.RandomState(SEED)
    # ---- known fragments from discovery tracks
    frags = []
    for i in range(N_FRAGS):
        L = LENGTHS[i % len(LENGTHS)]
        for _ in range(50):
            p = disc[rng.randint(0, len(disc))]
            if len(p["z"]) >= L + 2:
                s0 = rng.randint(0, len(p["z"]) - L)
                frags.append(
                    {"id": i, "L": L, "z": p["z"][s0 : s0 + L], "src": p["file"], "src_start": s0}
                )
                break

    # ---- implanted hosts: replace a random interval with the fragment
    def _splice(host_z, frag_z, pos):
        out = host_z.copy()
        n = len(frag_z)
        f = frag_z.copy()
        # 2-frame linear crossfades at both joins (identical treatment for
        # every condition; prevents discontinuity artifacts)
        for k in (0, 1):
            a = (k + 1) / 3.0
            f[k] = a * f[k] + (1 - a) * out[pos + k]
            f[n - 1 - k] = a * f[n - 1 - k] + (1 - a) * out[pos + n - 1 - k]
        out[pos : pos + n] = f
        return out

    implanted = []
    for f in frags:
        made = 0
        for _ in range(200):
            if made >= N_HOSTS_EACH:
                break
            h = val[rng.randint(0, len(val))]
            if len(h["z"]) < f["L"] + 6:
                continue
            pos = rng.randint(2, len(h["z"]) - f["L"] - 2)
            z = _splice(h["z"], f["z"], pos)
            implanted.append(
                {"frag": f["id"], "L": f["L"], "host": h["file"], "true_pos": pos, "z": z}
            )
            made += 1
    print(
        f"benchmark material: {len(frags)} known fragments, {len(implanted)} "
        f"implanted hosts, {len(val)} unmodified reference hosts",
        flush=True,
    )

    # ---- run frozen machinery
    m1_errs, m2_true_acc, m2_decoy_acc, m2_argmin_ok, m3_hits, m3_tot = [], [], [], [], 0, 0
    frag_by_id = {f["id"]: f for f in frags}
    for imp in implanted:
        f = frag_by_id[imp["frag"]]
        costs = {}
        for g in frags:
            c, s, e = std.subseq_dtw(g["z"], imp["z"])
            costs[g["id"]] = (c, s, e)
        c_true, s_true, e_true = costs[f["id"]]
        accepted_true = np.isfinite(c_true) and c_true <= FROZEN_RADIUS
        decoys = [costs[g["id"]][0] for g in frags if g["id"] != f["id"]]
        decoy_acc = [1.0 if (np.isfinite(c) and c <= FROZEN_RADIUS) else 0.0 for c in decoys]
        m2_true_acc.append(1.0 if accepted_true else 0.0)
        m2_decoy_acc.extend(decoy_acc)
        best_id = min(costs, key=lambda k: costs[k][0] if np.isfinite(costs[k][0]) else 1e12)
        m2_argmin_ok.append(1.0 if best_id == f["id"] else 0.0)
        m3_tot += 1
        m3_hits += 1 if accepted_true else 0
        if accepted_true:
            m1_errs.append(abs(s_true - imp["true_pos"]))
            m1_errs.append(abs(e_true - (imp["true_pos"] + f["L"])))
    # M4: unmodified-host acceptance (ordinary matchability reference)
    m4_rates = []
    for g in frags[:10]:
        hits = 0
        for h in val[:300]:
            c, _, _ = std.subseq_dtw(g["z"], h["z"])
            if np.isfinite(c) and c <= FROZEN_RADIUS:
                hits += 1
        m4_rates.append(hits / max(len(val[:300]), 1))

    m = {
        "M1_localization_median_abs_boundary_error_frames": round(
            float(np.median(m1_errs)) if m1_errs else None, 2
        ),
        "M1_n_boundary_errors": len(m1_errs),
        "M2_true_fragment_acceptance": round(float(np.mean(m2_true_acc)), 4),
        "M2_decoy_acceptance": round(float(np.mean(m2_decoy_acc)), 4),
        "M2_argmin_identity_correct": round(float(np.mean(m2_argmin_ok)), 4),
        "M3_occurrence_recovery": round(m3_hits / max(m3_tot, 1), 4),
        "M4_unmodified_host_reference_mean": round(float(np.mean(m4_rates)), 4),
    }
    # triage per review
    good_loc = (m["M1_localization_median_abs_boundary_error_frames"] or 99) <= 3
    good_group = (m["M2_true_fragment_acceptance"] - m["M2_decoy_acceptance"]) > 0.2
    sens = m["M3_occurrence_recovery"] >= 0.8
    if not sens:
        triage = "SENSITIVITY FAILURE: known positives are not recovered — the "
        "machinery cannot be used to adjudicate unit existence as configured"
    elif good_loc and good_group:
        triage = (
            "the frozen procedure demonstrated high recovery and "
            "localisation for the TESTED PLANTED fragments; this reduces "
            "concern about a gross failure of the exercised matching path, "
            "but does NOT identify why natural candidates failed the "
            "enrichment criterion and does not establish sensitivity to "
            "all naturally occurring variation (feature-level exact-copy "
            "insertion, known queries)"
        )
    else:
        triage = "known fragments recovered but grouping/localization weak — "
        "representation/matching limitation indicated"

    report = {
        "purpose": "known-fragment sensitivity + source-correspondence "
        "benchmark on constructed ground truth (validates the SEARCH, not "
        "the existence of biological units)",
        "validation_scope": {
            "insertion_level": "FEATURE-TRAJECTORY insertion (bypasses the "
            "source-to-feature measurement path; waveform insertion would "
            "exercise more of the pipeline)",
            "variation": "EXACT COPIES with 2-frame crossfades only — no "
            "independently specified transformations (timing/spectral/ "
            "background variation untested)",
            "query_supply": "KNOWN QUERIES SUPPLIED — matching and "
            "localisation are validated; seed selection and family "
            "construction (discovery) are NOT exercised by this benchmark",
            "denominators": "19 source fragments (not 171 fragment types); "
            "171 implanted hosts; decoys are the other known fragments",
        },
        "release_enforcement": enforcement,
        "frozen_radius": FROZEN_RADIUS,
        "material": {
            "known_fragments": len(frags),
            "implanted_hosts": len(implanted),
            "unmodified_reference_hosts": min(len(val), 300),
            "caller": 215,
        },
        "measures": m,
        "triage": triage,
        "notes": [
            "M2 decoys are the OTHER known fragments on the same implanted "
            "hosts — absorption of acoustically different fragments",
            "M4 is ordinary matchability (no implant) — the same quantity "
            "the caller-215 control distribution estimated",
            "source-correspondence review of the caller-215 candidate and "
            "control hits remains HUMAN work (source-reference panel)",
        ],
    }
    out = ROOT / "outputs" / "fragment_sensitivity_benchmark.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"measures": m, "triage": triage}, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
