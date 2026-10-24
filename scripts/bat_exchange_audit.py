#!/usr/bin/env python3
"""Bat interaction-reconstruction feasibility audit (advisor-directed
2026-10-20: the decisive prerequisite for event-level CAVEM).

QUESTION: can FileInfo.csv + Annotations.csv be joined into a
TRUSTWORTHY interaction timeline — and how many VALID DIRECTED
EXCHANGES survive prespecified continuity/identity gates?

R_exchange = N(valid directed exchanges) / N(candidate exchanges).

DATA (downloaded this session from the published Prat et al. 2017
release): FileInfo.csv (293,239 files: recording time, channel, voiced
sample spans), Annotations_orig.csv (91,080 rows keyed by FileID:
emitter, addressee, context, pre/post actions, start/end samples).
Local renamed audio (0.wav..) is row-aligned to the original order —
VERIFIED column-wise before any use (mismatch aborts).

GATES (prespecified; each reported as an attrition step):
  G1 ordering      events ordered by recording time WITHIN a
                   (treatment, channel) stream — simultaneous
                   same-time different-channel files are PARALLEL
                   captures, never sequential (no manufactured order);
  G2 session       continuity: consecutive files in-stream with
                   start-to-start gap <= 30 s belong to one session
                   (declared; 1 s timestamp resolution noted);
  G3 annotated     both consecutive events carry annotations;
  G4 identity      emitter and addressee valid (non-zero; 0 = unknown/
                   none per the release -> retained as UNCERTAIN, never
                   guessed into a response);
  G5 directed      candidate = E_i->R_i followed by any in-stream event
                   j>i with gap_ij <= G_MAX (5 s primary, 30 s
                   secondary); RESPONSE = caller_j == addressee_i;
                   self-continuation (caller_j == caller_i) is NOT a
                   response.
ENDPOINTS: R_exchange at each gate; session/caller/pair inventory;
response-latency distribution. MEASUREMENT STUDY ONLY — no model.

Usage:
    python scripts/bat_exchange_audit.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "egyptian_fruit_bats"
SESSION_GAP_S = 30.0
G_MAX_PRIMARY = 5.0
G_MAX_SECONDARY = 30.0


def _load():
    fi = {}
    with open(DATA / "FileInfo.csv") as fh:
        for row in csv.DictReader(fh):
            fid = int(row["FileID"])
            fi[fid] = {
                "t": datetime.strptime(row["Recording time"], "%Y-%m-%d %H:%M:%S"),
                "ch": int(row["Recording channel"]),
                "treat": int(row["Treatment ID"]),
                "dur_s": None,
            }
    ann = []
    with open(DATA / "Annotations_orig.csv") as fh:
        for row in csv.DictReader(fh):
            ann.append(
                {
                    "fid": int(row["FileID"]),
                    "emitter": int(row["Emitter"]),
                    "addressee": int(row["Addressee"]),
                    "context": int(row["Context"]),
                }
            )
    return fi, ann


def _verify_row_alignment():
    """Local annotations.csv (renamed 0.wav..) must match the original
    release row-for-row on shared columns, establishing the audio join."""
    with open(DATA / "annotations.csv") as fh:
        loc = list(csv.DictReader(fh))
    with open(DATA / "Annotations_orig.csv") as fh:
        orig = list(csv.DictReader(fh))
    assert len(loc) == len(orig), (len(loc), len(orig))
    mism = 0
    for a, b in zip(loc, orig):
        if (a["Emitter"], a["Addressee"], a["Context"]) != (
            b["Emitter"],
            b["Addressee"],
            b["Context"],
        ):
            mism += 1
    return {"n_rows": len(loc), "column_mismatches": mism, "aligned": mism == 0}


def main() -> None:
    fi, ann = _load()
    align = _verify_row_alignment()
    print(f"row alignment: {align}", flush=True)
    if not align["aligned"]:
        print("ALIGNMENT FAILED — aborting (audio join unverifiable)", flush=True)
        return

    # in-stream event lists keyed by (treatment, channel)
    streams = defaultdict(list)
    n_missing_fi = 0
    for a in ann:
        f = fi.get(a["fid"])
        if f is None:
            n_missing_fi += 1
            continue
        a["t"] = f["t"]
        streams[(f["treat"], f["ch"])].append(a)
    for k in streams:
        streams[k].sort(key=lambda x: x["t"])

    # sessions per stream (G2) and gate attrition
    report = {"alignment": align, "n_annotations": len(ann), "n_missing_fileinfo": n_missing_fi}
    inv = {"streams": len(streams), "sessions": 0, "callers": set(), "pairs": set()}
    g = {
        "G3_annotated_pairs": 0,
        "G4_identity_valid": 0,
        "G5_candidates_5s": 0,
        "responses_5s": 0,
        "candidates_30s": 0,
        "responses_30s": 0,
    }
    latencies = []
    for (treat, ch), evs in streams.items():
        # session boundaries
        sess, cur = [], [evs[0]]
        for e in evs[1:]:
            gap = (e["t"] - cur[-1]["t"]).total_seconds()
            if gap > SESSION_GAP_S:
                sess.append(cur)
                cur = [e]
            else:
                cur.append(e)
        sess.append(cur)
        inv["sessions"] += len(sess)
        for s in sess:
            for i in range(len(s)):
                e = s[i]
                inv["callers"].add(e["emitter"])
                if e["addressee"]:
                    inv["pairs"].add((e["emitter"], e["addressee"]))
            for i in range(len(s) - 1):
                a, b = s[i], s[i + 1]
                g["G3_annotated_pairs"] += 1
                if a["emitter"] and a["addressee"] and b["emitter"]:
                    g["G4_identity_valid"] += 1
                    gap = (b["t"] - a["t"]).total_seconds()
                    if gap <= G_MAX_SECONDARY:
                        g["candidates_30s"] += 1
                        if b["emitter"] == a["addressee"] and b["emitter"] != a["emitter"]:
                            g["responses_30s"] += 1
                            latencies.append(gap)
                        if gap <= G_MAX_PRIMARY:
                            g["G5_candidates_5s"] += 1
                            if b["emitter"] == a["addressee"] and b["emitter"] != a["emitter"]:
                                g["responses_5s"] += 1
    import numpy as np

    out = {
        **report,
        **{k: v for k, v in g.items()},
        "inventory": {
            "streams": inv["streams"],
            "sessions": inv["sessions"],
            "distinct_callers": len(inv["callers"]),
            "distinct_directed_pairs": len(inv["pairs"]),
        },
        "R_exchange_5s": round(g["responses_5s"] / max(g["G5_candidates_5s"], 1), 4),
        "R_exchange_30s": round(g["responses_30s"] / max(g["candidates_30s"], 1), 4),
        "response_latency_s": {
            "median": round(float(np.median(latencies)), 2) if latencies else None,
            "p90": round(float(np.percentile(latencies, 90)), 2) if latencies else None,
            "n": len(latencies),
        },
        "gates": "G1 order within (treatment, channel) streams; parallel "
        "same-time channels never sequenced; G2 session gap <= 30 s; G3 "
        "both annotated; G4 non-zero identities (0 retained as "
        "UNCERTAIN, excluded here, never guessed); G5 response = next "
        "in-stream caller == addressee, self-continuation excluded",
        "framing": "measurement study only; a positive R_exchange "
        "justifies event-level CAVEM on validated sequences; no model "
        "trained here; no meaning claims",
    }
    (ROOT / "outputs" / "bat_exchange_audit.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps({k: out[k] for k in ("R_exchange_5s", "R_exchange_30s", "inventory")}, indent=1),
        flush=True,
    )
    print(json.dumps(g, indent=1), flush=True)
    print("wrote outputs/bat_exchange_audit.json", flush=True)


if __name__ == "__main__":
    main()
