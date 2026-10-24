#!/usr/bin/env python3
"""Individual acoustic-repertoire study (one caller, two parallel outputs).

QUESTION (per review): which complete acoustic forms does animal A repeat,
and which smaller fragments recur within its DIFFERENT vocalisations — with
discovery, control calibration AND validation all conditional on the same
individual, and cross-caller similarity explicitly NOT a requirement.

PREREGISTRATION (frozen before the search):

  CALLER SELECTION: by metadata only — the caller with the most independent
  recording blocks among eligible parents (tie-break: more events), chosen
  before any match score is computed.

  PARTITIONS: that caller's recording blocks hash-split 40/20/40 into
  discovery / calibration / validation. All events of one block stay
  together. Calibration material sets thresholds; validation is scored once.

  SCALING: the frozen COMMON corpus scaling (fitted on the pooled discovery
  partition, exactly as the prior studies) is retained — this isolates the
  effect of restricting the search POPULATION (§4 of the review).

  ARM A (repeated whole vocalisations): whole-call vectors; radius R_wc
  frozen at WC_Q-th percentile of CALIBRATION within-caller NN distances;
  densest-first radius-to-exemplar families (>= MIN_FAMILY events);
  validation = acceptance of the caller's independent calls; control =
  random whole-call queries from the caller's discovery material under the
  identical rule.

  ARM B (recurrent internal fragments): within-caller seeds at declared
  lengths {8,16,24}; subsequence DTW (free ends, degeneracy guard);
  radius frozen from the CALLER'S OWN calibration random-fragment cost
  distribution (CTRL_PCT percentile); exemplar-star families (>= 3 distinct
  events); validation searches ONLY the caller's independent hosts
  (restriction applied BEFORE neighbour selection — a post-filter of an
  unrestricted search could miss the best eligible same-caller match);
  control = the caller's own random discovery fragments, duration-matched
  by construction, identical procedure.

  RELATION: each fragment is classified by whether its occurrence HOSTS are
  one repeated whole-call form (all pairwise whole-call cosine >= COS_ONE)
  or several acoustically different calls — a stable component of a form vs
  a candidate reusable intra-call unit.

  SUCCESS TABLE (§7): complete calls recur across independent groups ->
  supported repeated whole-call form; fragment within one form only ->
  repeatable component; fragment across different calls -> candidate
  reusable unit; discovery/one-episode only -> persistence not established;
  recovery not exceeding the declared reference -> no demonstrated
  transferable unit under this procedure.

  Controls are REFERENCE BASELINES, not automatic false-positive rates
  (random fragments can themselves contain recurrent material).

  FRAMING: recurrence within one animal's repertoire. Uniqueness to the
  individual is NOT claimed (that needs other animals' recordings — a
  later, optional question). No words, no meaning.

Usage:
    python scripts/individual_repertoire.py --self-test
    python scripts/individual_repertoire.py

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

WC_Q = 25.0
CTRL_PCT = 5.0
MIN_FAMILY = 3
LENGTHS = (8, 16, 24)
COS_ONE = 0.85
SEED = 0
DISC_PCT, CAL_PCT = 40, 20  # remainder -> validation


def prereg_spec():
    return {
        "design": "individual repertoire: repeated whole calls + recurrent "
        "internal fragments, one caller, three frozen partitions",
        "caller_selection": "max independent recording blocks (metadata only)",
        "partitions": f"block hash {DISC_PCT}/{CAL_PCT}/{100 - DISC_PCT - CAL_PCT} "
        "discovery/calibration/validation",
        "scaling": "frozen common corpus scaling (population restriction "
        "isolated from geometry changes)",
        "arm_a": f"whole-call radius = WC_Q {WC_Q}% of calibration NN dists; "
        "densest-first exemplar families",
        "arm_b": f"fragment radius = CTRL_PCT {CTRL_PCT}% of caller-own "
        "calibration random-fragment costs; subsequence DTW, free ends",
        "controls": "caller-own random queries (whole calls / fragments), "
        "duration-matched, identical procedure — reference baselines",
        "success_table": [
            "repeated whole-call form across groups",
            "repeatable component of a form",
            "candidate reusable intra-call unit",
            "discovery-local only",
            "no demonstrated transferable unit",
        ],
    }


def _stage(block: str) -> str:
    h = int(hashlib.sha1(block.encode()).hexdigest()[:8], 16) % 100
    return (
        "discovery" if h < DISC_PCT else ("calibration" if h < DISC_PCT + CAL_PCT else "validation")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    _real_run()


def _self_test() -> None:
    """Synthetic one-caller repertoire: 2 repeated forms + shared fragment."""
    import numpy as np

    import subcall_term_discovery as std

    rng = np.random.RandomState(0)
    formA = rng.randn(20, 12) * 0.4 + np.sin(np.linspace(0, 6, 20))[:, None]
    formB = rng.randn(16, 12) * 0.4 + np.cos(np.linspace(0, 4, 16))[:, None]
    frag = rng.randn(10, 12) * 0.3 + np.sin(np.linspace(0, 3, 10))[:, None]
    events = []
    for rep in range(3):  # whole-form repeats
        events.append((f"wa{rep}", formA + 0.05 * rng.randn(20, 12)))
        events.append((f"wb{rep}", formB + 0.05 * rng.randn(16, 12)))
    for i in range(5):  # fragment inside DIFFERENT-form hosts
        host = rng.randn(14 + 2 * i, 12) * 0.8
        pos = 2 + i
        events.append(
            (f"fx{i}", np.vstack([host[:pos], frag + 0.05 * rng.randn(10, 12), host[pos:]]))
        )
    tracks = {n: tr for n, tr in events}
    allf = np.vstack([tr for _, tr in events])
    mu, sd = allf.mean(0), allf.std(0) + 1e-9
    z = {n: (tr - mu) / sd for n, tr in tracks.items()}
    frag_z = (frag - mu) / sd
    # fragment recovered inside different-form hosts
    costs = [std.subseq_dtw(frag_z, z[n])[0] for n, _ in events if n.startswith("fx")]
    assert max(costs) < 0.6, costs
    # whole-form correspondence
    ca = std.subseq_dtw(z["wa0"], z["wa2"])[0]
    cab = std.subseq_dtw(z["wa0"], z["wb2"])[0]
    assert ca < cab, (ca, cab)
    print(
        f"SELF-TEST OK (fragment max cost {max(costs):.3f}; formA-vs-A "
        f"{ca:.3f} < formA-vs-B {cab:.3f})",
        flush=True,
    )


def _real_run() -> None:
    import numpy as np

    import subcall_term_discovery as std
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        FEATURE_SCHEMA_VERSION,
    )
    from bioacoustic_pipeline.pipeline.release_contract import (
        MaskProvenance,
        ScientificRunContract,
        raise_for_science,
        source_manifest_from,
    )

    parents, feats = std.load_parents()
    # ---- caller selection: METADATA ONLY (blocks, then events)
    stat: dict = {}
    for p in parents:
        s = stat.setdefault(p["caller"], {"blocks": set(), "events": 0})
        s["blocks"].add(p["block"])
        s["events"] += 1
    target = max(stat, key=lambda y: (len(stat[y]["blocks"]), stat[y]["events"]))
    mine = [p for p in parents if p["caller"] == target]
    blocks_t = sorted({p["block"] for p in mine})
    print(
        f"caller {target}: {len(mine)} eligible events across {len(blocks_t)} "
        f"recording blocks (metadata-selected before any score)",
        flush=True,
    )

    for p in parents:
        p["stage"] = _stage(p["block"]) if p["caller"] == target else "other"
    parts = {
        s: [p for p in mine if p["stage"] == s] for s in ("discovery", "calibration", "validation")
    }
    for s, v in parts.items():
        print(f"  {s}: {len(v)} events / {len({p['block'] for p in v})} blocks", flush=True)

    files = sorted({p["file"] for p in mine})
    contract = ScientificRunContract(
        run_name="individual_repertoire",
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
                produced_by="valid spans >= 8 frames",
                win_ms=40.0,
                hop_ms=10.0,
                amplitude_stage="native waveform",
            ),
        },
        source_manifest=source_manifest_from(
            files,
            [next(p["stage"] for p in mine if p["file"] == f) for f in files],
            [next(p["block"] for p in mine if p["file"] == f) for f in files],
        ),
        eval_split={
            "provenance": "block-hash 40/20/40 within one caller",
            "unit": "recording block",
        },
    )
    enforcement = raise_for_science(contract)

    # frozen COMMON scaling (pooled discovery partition, as prior studies)
    pooled_disc = [p for p in parents if std.part_of_block(p["block"]) == "discovery"]
    allf = np.vstack([p["track"] for p in pooled_disc])
    mu, sd = allf.mean(0), allf.std(0) + 1e-9
    for p in parents:
        p["z"] = (p["track"] - mu) / sd
        p["vec"] = p["track"].mean(0)

    rng = np.random.RandomState(SEED)

    # ================= ARM A: repeated whole vocalisations
    cal = parts["calibration"]
    if len(cal) >= 4:
        m = np.stack([p["vec"] for p in cal])
        d = np.linalg.norm(m[:, None] - m[None], axis=2) + np.eye(len(m)) * 1e9
        r_wc = float(np.percentile(d.min(axis=1), WC_Q))
    else:
        r_wc = None
    arm_a = {"radius": r_wc, "families": [], "control": None}
    if r_wc:
        disc = sorted(parts["discovery"], key=lambda p: (p["file"], p["start"]))
        remaining = list(disc)
        while len(remaining) >= MIN_FAMILY:
            vecs = np.stack([p["vec"] for p in remaining])
            seed = 0
            dd = np.linalg.norm(vecs - vecs[seed], axis=1)
            members = [remaining[i] for i in np.where(dd <= r_wc)[0]]
            if len(members) >= MIN_FAMILY:
                arm_a["families"].append(
                    {
                        "exemplar": members[0]["file"],
                        "exemplar_interval": [members[0]["start"], members[0]["end"]],
                        "n_events": len(members),
                        "n_files": len({x["file"] for x in members}),
                        "n_blocks": len({x["block"] for x in members}),
                        "contexts": sorted({x["context"] for x in members}),
                    }
                )
            gone = {id(x) for x in members}
            remaining = [p for p in remaining if id(p) not in gone]
        # validation: caller's independent calls
        val = parts["validation"]
        for f in arm_a["families"]:
            ex = next(p for p in parts["discovery"] if p["file"] == f["exemplar"])
            f["validation_hits"] = sum(
                1 for p in val if float(np.linalg.norm(p["vec"] - ex["vec"])) <= r_wc
            )
            f["validation_n"] = len(val)
            f["validation_rate"] = round(f["validation_hits"] / max(len(val), 1), 4)
        # control: random discovery whole-call queries, identical rule
        ctrl = []
        for i in range(25):
            q = disc[rng.randint(0, len(disc))]
            ctrl.append(
                sum(1 for p in val if float(np.linalg.norm(p["vec"] - q["vec"])) <= r_wc)
                / max(len(val), 1)
            )
        arm_a["control"] = {
            "mean_rate": round(float(np.mean(ctrl or [0])), 4),
            "max_rate": round(float(np.max(ctrl or [0])), 4),
            "n": len(ctrl),
        }

    # ================= ARM B: recurrent internal fragments
    disc_ev = sorted(parts["discovery"], key=lambda p: (p["file"], p["start"]))
    cal_ev = parts["calibration"]
    r_frag = None
    if len(cal_ev) >= 6:
        costs = []
        for i in range(40):
            p = cal_ev[rng.randint(0, len(cal_ev))]
            L = LENGTHS[i % len(LENGTHS)]
            if len(p["z"]) < L + 2:
                continue
            s0 = rng.randint(0, len(p["z"]) - L)
            for k2 in rng.randint(0, len(cal_ev), size=6):
                p2 = cal_ev[k2]
                if p2["file"] == p["file"]:
                    continue
                costs.append(std.subseq_dtw(p["z"][s0 : s0 + L], p2["z"])[0])
        costs = [c for c in costs if np.isfinite(c)]
        if costs:
            r_frag = float(np.percentile(costs, CTRL_PCT))
    arm_b = {"radius": r_frag, "families": [], "control": None}
    if r_frag:
        seeds = []
        for L in LENGTHS:
            for qi, pa in enumerate(disc_ev):
                wq, sq = std._windows(pa["z"], L, L)
                if not len(wq):
                    continue
                for ci, pb in enumerate(disc_ev):
                    if qi == ci or not (len(pb["z"]) >= L):
                        continue
                    wc, sc = std._windows(pb["z"], L, 2)
                    if not len(wc):
                        continue
                    sim = wq @ wc.T
                    dm = np.sqrt(np.maximum(2 - 2 * sim, 0))
                    best = dm.min(axis=1)
                    bj = dm.argmin(axis=1)
                    i_row = int(np.argmin(best))
                    dmin = float(best[i_row])
                    j = int(bj[i_row])
                    if dmin <= r_frag:
                        seeds.append(
                            {
                                "L": L,
                                "dist": dmin,
                                "a": pa,
                                "a_s": sq[i_row],
                                "b": pb,
                                "b_s": sc[j],
                            }
                        )
        seeds.sort(key=lambda s: s["dist"])
        used: set = set()
        for s in seeds[:120]:
            if s["a"]["file"] == s["b"]["file"]:
                continue
            key = (s["a"]["file"], s["a_s"], s["b"]["file"], s["b_s"])
            if key in used:
                continue
            q = s["a"]["z"][s["a_s"] : s["a_s"] + s["L"]]
            cost, cs, ce = std.subseq_dtw(q, s["b"]["z"])
            if not np.isfinite(cost) or cost > r_frag:
                continue
            occ = [(s["a"], s["a_s"], s["a_s"] + s["L"]), (s["b"], cs, ce)]
            for p in disc_ev:
                if p["file"] in {s["a"]["file"], s["b"]["file"]}:
                    continue
                c2, s2, e2 = std.subseq_dtw(q, p["z"])
                if np.isfinite(c2) and c2 <= r_frag:
                    occ.append((p, s2, e2))
            if len({o[0]["file"] for o in occ}) < MIN_FAMILY:
                continue
            used.add(key)
            # relation to whole-call forms: host pairwise cosine
            hv = np.stack([o[0]["vec"] for o in occ])
            hv = hv / (np.linalg.norm(hv, axis=1, keepdims=True) + 1e-9)
            cos = hv @ hv.T
            one_form = (
                bool(np.triu(cos, 1)[np.triu(cos, 1) > -2].min() >= COS_ONE)
                if len(occ) > 1
                else True
            )
            # validation: ONLY the caller's independent hosts (before selection)
            vhits = []
            for p in parts["validation"]:
                c3, s3, e3 = std.subseq_dtw(q, p["z"])
                if np.isfinite(c3) and c3 <= r_frag:
                    vhits.append(
                        {"file": p["file"], "interval": [int(s3), int(e3)], "cost": round(c3, 4)}
                    )
            arm_b["families"].append(
                {
                    "exemplar": s["a"]["file"],
                    "exemplar_interval": [int(s["a_s"]), int(s["a_s"] + s["L"])],
                    "L": s["L"],
                    "n_events": len({o[0]["file"] for o in occ}),
                    "n_blocks": len({o[0]["block"] for o in occ}),
                    "occurrence_intervals": [
                        {
                            "file": o["file"],
                            "interval": [int(a), int(b)],
                            "context": o["context"],
                        }
                        for (o, a, b) in occ[:12]
                    ],
                    "hosts_all_one_whole_form": one_form,
                    "min_host_pairwise_cos": round(float(np.triu(cos, 1).min()), 4),
                    "validation_hits": vhits,
                    "validation_n": len(parts["validation"]),
                    "validation_rate": round(len(vhits) / max(len(parts["validation"]), 1), 4),
                }
            )
            if len(arm_b["families"]) >= 30:
                break
        # control: caller-own random fragments, duration-matched, identical
        ctrl = []
        for i in range(25):
            p = disc_ev[rng.randint(0, len(disc_ev))]
            L = LENGTHS[i % len(LENGTHS)]
            if len(p["z"]) < L + 2:
                continue
            s0 = rng.randint(0, len(p["z"]) - L)
            q = p["z"][s0 : s0 + L]
            hits = sum(
                1
                for pv in parts["validation"]
                if np.isfinite(std.subseq_dtw(q, pv["z"])[0])
                and std.subseq_dtw(q, pv["z"])[0] <= r_frag
            )
            ctrl.append(hits / max(len(parts["validation"]), 1))
        arm_b["control"] = {
            "mean_rate": round(float(np.mean(ctrl or [0])), 4),
            "max_rate": round(float(np.max(ctrl or [0])), 4),
            "n": len(ctrl),
        }

    # ---- success table
    table = {
        "repeated_whole_call_forms": [
            f
            for f in arm_a["families"]
            if f["n_blocks"] >= 2
            and (f["validation_rate"] or 0)
            > (arm_a["control"]["max_rate"] if arm_a["control"] else 1)
        ],
        "repeatable_components_of_one_form": [
            f for f in arm_b["families"] if f["hosts_all_one_whole_form"]
        ],
        "candidate_reusable_intra_call_units": [
            f
            for f in arm_b["families"]
            if not f["hosts_all_one_whole_form"]
            and f["n_blocks"] >= 2
            and (f["validation_rate"] or 0)
            > (arm_b["control"]["max_rate"] if arm_b["control"] else 1)
        ],
        "discovery_local_only": [
            f["exemplar"] for f in arm_a["families"] + arm_b["families"] if f["n_blocks"] < 2
        ],
        "no_demonstrated_transfer": [
            f["exemplar"]
            for f in arm_a["families"] + arm_b["families"]
            if (f.get("validation_rate") or 0)
            <= ((arm_a["control"] or arm_b["control"] or {}).get("max_rate", 1))
        ],
    }
    report = {
        "preregistration": prereg_spec(),
        "release_enforcement": enforcement,
        "caller": {
            "id": int(target),
            "events": len(mine),
            "blocks": len(blocks_t),
            "selection": "metadata only (blocks, then events)",
        },
        "partitions": {
            s: {"events": len(v), "blocks": len({p["block"] for p in v})} for s, v in parts.items()
        },
        "arm_a_whole_calls": arm_a,
        "arm_b_fragments": arm_b,
        "success_table": {
            k: (
                v
                if isinstance(v, list) and v and isinstance(v[0], str)
                else [x.get("exemplar") if isinstance(x, dict) else x for x in v]
            )
            for k, v in table.items()
        },
        "controls_note": "reference baselines, not automatic false-positive "
        "rates — random fragments can contain recurrent material",
        "scope": "recurrence within ONE animal's repertoire; uniqueness NOT "
        "claimed; contexts are metadata; no words, no meaning",
    }
    out = ROOT / "outputs" / "individual_repertoire.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                "caller": report["caller"],
                "partitions": report["partitions"],
                "arm_a": {
                    "radius": arm_a["radius"],
                    "families": len(arm_a["families"]),
                    "control": arm_a["control"],
                },
                "arm_b": {
                    "radius": arm_b["radius"],
                    "families": len(arm_b["families"]),
                    "control": arm_b["control"],
                },
                "success_table": {k: len(v) for k, v in table.items()},
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
