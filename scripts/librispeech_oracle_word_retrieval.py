#!/usr/bin/env python3
"""Speech-side oracle-boundary word retrieval — the missing positive-control
comparison (advisor-directed Phase A, 2026-10-20).

QUESTION: does the frozen fingerprint + fixed-endpoint DTW discriminate
naturally spoken WORDS when their boundaries are SUPPLIED? This completes
the symmetry with Bengalese Stage B and interprets the free-start
word-recovery null (Recall@1 0.020): a low free-start score does not show
the representation cannot discriminate words at known locations.

DESIGN (prespecified to mirror librispeech_word_recovery exactly):
same exact-match corpus (2534 utterances), same speaker-blocked split,
same 30 word types (15 high + 15 mid, RandomState(0)) and same 5
median-duration query tokens per type, same duration/frequency-matched
control types. The ONLY change: query and pool segments are the TRUE
MFA word spans (window-center-in-word frame mapping) and matching is
fixed-endpoint DTW ranking (no search, no thresholds) over ALL eval
word segments (cross-speaker by construction).

ENDPOINTS: Recall@1 / Recall@10 same-word (query-token clustered
bootstrap B=1000) for target types and matched controls; majority-class
prevalence chance. DECISION RULE: representation "discriminates words at
known locations" if Recall@1 >= 0.5 and >= 2x control; "limited" if < 2x
control. Caveat carried: MFA-reference, human-validation pending.

Usage:
    python scripts/librispeech_oracle_word_retrieval.py

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

from bengalese_score_diagnostics import dtw_full
from librispeech_word_recovery import (
    CACHE,
    MIN_FRAMES,
    N_HIGH,
    N_MID,
    N_TOK,
    SEED,
    SR,
    _load_corpus,
)
from mfa_alignment_qa import LS_DIR, TG_DIR, parse_textgrid
from natural_unit_reference_benchmark import _boot_rate

WIN_MS, HOP_MS = 40.0, 10.0
BOOT_B = 1000
MIN_EVAL_OCC = 10

_POOL: list = []


def _query_job(q):
    """Module-level (picklable): one query token ranked against the pool."""
    import numpy as np

    costs = []
    for j, v in enumerate(_POOL):
        if v["uid"] == q["uid"]:
            continue
        c = dtw_full(q["seg"], v["seg"])
        if np.isfinite(c):
            costs.append((c, j))
    if not costs:
        return None
    costs.sort(key=lambda r: r[0])
    w = q["label"]
    r1 = 1 if _POOL[costs[0][1]]["label"] == w else 0
    r10 = 1 if any(_POOL[j]["label"] == w for _, j in costs[:10]) else 0
    return r1, r10


def _word_segments(u, traj, hop, win):
    """Supported MFA word spans -> fingerprint segments for one utterance."""
    import numpy as np

    tr, valid = traj
    n = len(tr)
    centers = np.arange(n) * hop + win / 2.0
    out, n_unsup = [], 0
    for a, b, w in u["words"]:
        idx = np.where((centers >= a * SR) & (centers < b * SR))[0]
        if len(idx) < MIN_FRAMES:
            n_unsup += 1
            continue
        s, e = int(idx[0]), int(idx[-1]) + 1
        seg = tr[s:e]
        if np.all(seg == 0):
            continue
        out.append({"label": w.lower(), "seg": seg.astype(np.float32), "uid": u["uid"]})
    return out, n_unsup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    import numpy as np

    hop, win = int(SR * HOP_MS / 1000), int(SR * WIN_MS / 1000)
    utts = _load_corpus()
    for u in utts:
        tg = (
            TG_DIR / Path(u["flac"]).relative_to(LS_DIR).parent / f"{Path(u['flac']).stem}.TextGrid"
        )
        tiers = parse_textgrid(tg)
        u["words"] = [(a, b, w) for a, b, w in tiers.get("words", []) if w.strip()]

    d = np.load(CACHE, allow_pickle=False)
    names = list(d["names"])
    off = d["offsets"]
    # FIX (2026-10-20, the OOM root cause): NpzFile.__getitem__ does NOT
    # cache members — accessing d["frames"]/d["valid"] per utterance
    # re-decompresses and each slice-view retains its own copy (the 57 GB
    # blowup and the 6 GB-cap kill). Materialise each member ONCE, then
    # slice; peak RSS drops to ~0.3 GB.
    frames = d["frames"]
    valid = d["valid"]
    traj = {
        n: (frames[off[i] : off[i + 1]], valid[off[i] : off[i + 1]]) for i, n in enumerate(names)
    }
    del d

    fit = [u for u in utts if u["part"] == "fit"]
    ev = [u for u in utts if u["part"] == "eval"]
    pool, n_unsup_total = [], 0
    for u in ev:
        segs, n_unsup = _word_segments(u, traj[u["uid"]], hop, win)
        pool.extend(segs)
        n_unsup_total += n_unsup
    print(
        f"eval word segments (pool): {len(pool)} from {len(ev)} utterances "
        f"({n_unsup_total} unsupported short words excluded)",
        flush=True,
    )

    eval_occ, eval_dur = {}, {}
    for s in pool:
        eval_occ[s["label"]] = eval_occ.get(s["label"], 0) + 1
        eval_dur.setdefault(s["label"], []).append(len(s["seg"]))
    eligible = sorted(w for w, n_v in eval_occ.items() if n_v >= MIN_EVAL_OCC)
    rng = np.random.RandomState(SEED)
    high = sorted(eligible, key=lambda w: -eval_occ[w])[:N_HIGH]
    mid_pool = [w for w in eligible if MIN_EVAL_OCC <= eval_occ[w] <= 49 and w not in high]
    mid = sorted(rng.choice(mid_pool, size=min(N_MID, len(mid_pool)), replace=False).tolist())
    types = high + mid

    controls = {}
    for w in types:
        dw = float(np.median(eval_dur[w]))
        fw = np.log(eval_occ[w])
        cands = [
            c
            for c in eligible
            if c != w
            and abs(np.median(eval_dur[c]) - dw) <= 0.25 * dw
            and abs(np.log(eval_occ[c]) - fw) <= 0.5
        ]
        controls[w] = sorted(cands)[0] if cands else None

    fit_tok = {}
    for u in fit:
        segs, _ = _word_segments(u, traj[u["uid"]], hop, win)
        for s in segs:
            fit_tok.setdefault(s["label"], []).append(s)

    def _pick(w, n):
        toks = fit_tok.get(w, [])
        if not toks:
            return []
        durs = np.array([len(t["seg"]) for t in toks])
        med = float(np.median(durs))
        order = np.argsort(np.abs(durs - med))
        return [toks[i] for i in order[:n]]

    def _search(q):
        costs = []
        for j, v in enumerate(pool):
            if v["uid"] == q["uid"]:
                continue
            c = dtw_full(q["seg"], v["seg"])
            if np.isfinite(c):
                costs.append((c, j))
        costs.sort(key=lambda r: r[0])
        return costs

    def _arm(sel_types, arm_name):
        # SINGLE-PROCESS by design: the earlier fork-parallel variant was
        # OOM-killed (12 forked workers each touching inherited pages; kernel
        # logged 57 GB anon RSS). Sequential DTW over the pool needs ~1 GB
        # and finishes in minutes — memory safety over speed (results
        # identical; _query_job retained for reference). Results are
        # checkpointed per query so an interrupted run resumes losslessly.
        import json as _json

        ckpt = ROOT / "cache" / "libriword" / f"oracle_ckpt_{arm_name}.jsonl"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        done = {}
        if ckpt.exists():
            for line in ckpt.read_text().splitlines():
                if line.strip():
                    r = _json.loads(line)
                    done[(r["w"], r["uid"], r["len"])] = (r["r1"], r["r10"])
        per_tok = []
        jobs = [(w, q) for w in sel_types for q in _pick(w, N_TOK)]
        with open(ckpt, "a") as fh:
            for k, (w, q) in enumerate(jobs):
                key = (w, q["uid"], len(q["seg"]))
                if key in done:
                    per_tok.append(done[key])
                    continue
                costs = _search(q)
                if not costs:
                    continue
                r1 = 1 if pool[costs[0][1]]["label"] == w else 0
                r10 = 1 if any(pool[j]["label"] == w for _, j in costs[:10]) else 0
                per_tok.append((r1, r10))
                fh.write(
                    _json.dumps({"w": w, "uid": q["uid"], "len": key[2], "r1": r1, "r10": r10})
                    + "\n"
                )
                fh.flush()
                if (k + 1) % 25 == 0:
                    import resource

                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                    print(
                        f"  [{arm_name}] {k + 1}/{len(jobs)} queries (peak RSS {rss:.1f} GB)",
                        flush=True,
                    )
        return {
            "n_queries": len(per_tok),
            "recall1": _boot_rate([(a, 1) for a, _ in per_tok], b=BOOT_B) if per_tok else None,
            "recall10": _boot_rate([(b, 1) for _, b in per_tok], b=BOOT_B) if per_tok else None,
        }

    import gc

    del traj, utts, fit
    gc.collect()
    print(
        f"memory-optimised: trajectory cache freed before search; pool {len(pool)} segs", flush=True
    )
    target = _arm(types, "target")
    ctrl_types = sorted({c for c in controls.values() if c})
    ctrl = _arm(ctrl_types, "control") if ctrl_types else None
    chance = max(eval_occ.values()) / len(pool) if pool else None
    mean_prev = float(np.mean([eval_occ[w] / len(pool) for w in types]))

    r1 = target["recall1"]["point"] if target["recall1"] else None
    c1 = ctrl["recall1"]["point"] if ctrl and ctrl["recall1"] else None
    verdict = {
        "discriminates_words_at_known_locations": bool(
            r1 is not None and r1 >= 0.5 and (c1 is None or r1 >= 2 * c1)
        ),
        "limited": bool(r1 is not None and c1 is not None and r1 < 2 * c1),
    }

    report = {
        "study": "speech oracle-boundary word retrieval (Phase A: the "
        "missing positive-control comparison for the free-start null)",
        "reference_provenance": "MFA machine alignments — human-validation "
        "pending (outputs/mfa_alignment_review/)",
        "design": "mirrors librispeech_word_recovery exactly (types, "
        "tokens, controls, split); only the matching is fixed-endpoint "
        "DTW on TRUE word spans",
        "pool": {"n_eval_segments": len(pool), "unsupported_excluded": n_unsup_total},
        "target": target,
        "matched_control": ctrl,
        "chance": {
            "majority_class": round(chance, 4) if chance else None,
            "mean_type_prevalence": round(mean_prev, 4),
        },
        "verdicts": verdict,
        "stop_rule": "one run",
    }
    out = ROOT / "outputs" / "librispeech_oracle_word_retrieval.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {k: report[k] for k in ("target", "matched_control", "chance", "verdicts")}, indent=1
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
