#!/usr/bin/env python3
"""LibriSpeech word-recovery benchmark — label-blind discovery vs known words.

PREREGISTRATION (frozen before any score is computed; advisor-directed
2026-10-20). Reference: public MFA alignments (Zenodo 2619474, CC-BY-4.0,
md5-verified). CAVEAT CARRIED ON EVERY RESULT: the timestamps are MACHINE
alignments; the deterministic human-validation sample
(outputs/mfa_alignment_review/) is PENDING — all results are
"MFA-reference, human-validation pending".

  "Labels remain hidden during the acoustic search and are used only at
   scoring time."

MATERIAL: dev-clean (2703 flacs); only utterances whose TextGrid token
sequence exactly matches the transcript (2534; the 169 exclusions are
tokenization-level, e.g. possessive clitics — QA in
outputs/mfa_alignment_qa.json). Words shorter than the window contract
(<3 frames at 40ms/10ms) are UNSUPPORTED: reported, excluded from query
selection, retained in availability denominators.

SPLITS (speaker-blocked, deterministic): sha1(speaker)%100<70 -> fit
(query source); else eval (search hosts). Cross-speaker recovery is
therefore the DEFAULT arm (host speakers unseen). A separate SAME-SPEAKER
arm scores queries from eval speakers against their own OTHER utterances.

QUERY SELECTION (prespecified): word types with >=10 supported eval
occurrences; 15 highest-frequency + 15 seeded-random mid-frequency
(10-49 eval occurrences, RandomState(0)); 5 query tokens per type from
fit utterances — the five tokens nearest the type's MEDIAN fit duration
(deterministic). Matched controls: per type, one control type matched on
median duration (+-25%) and log eval frequency (+-0.5), searched with
identical machinery.

SEARCH (frozen, label-blind, NO acceptance threshold — pure ranking,
which sidesteps the acceptance-calibration failure chain): 12-feature
fingerprint trajectories (schema v2, species human, 40ms/10ms,
window-center-in-word frame mapping); per host kbest_candidates (k=5,
the deployed top-K mechanism); candidates pooled globally by cost.

ENDPOINTS (decision rules fixed now):
  E1 Recall@1 (top-1 candidate overlaps a same-word token, IoU>=0.5),
     cross-speaker. Tier "recovery demonstrated" if >=0.2 AND >=2x the
     matched-control Recall@1; "absent" if <0.05 or <2x control.
  E2 Recall@5, Recall@10 (any correct candidate in the global top-K).
  E3 boundary localisation among correct top-1s: median IoU, onset/offset
     error (ms); complete-miss fraction = top-1 with no same-word overlap
     at all. Tier "adequate" if median IoU >= 0.5.
  E4 contextual reuse (H3): among correct top-10 candidates, fraction in
     a different (prev,next) word context than the query, vs the type's
     reference context diversity (1 - Simpson over eval contexts).
     Tier "context-neutral" if |obs - ref| <= 0.15.
  E5 speaker contrast: same-speaker vs cross-speaker Recall@1.
  Availability floor: mean over hosts of (#type tokens / #word tokens).
UNCERTAINTY: query-token clustered bootstrap (B=1000) for E1/E2.

STOP RULE: one run; no parameter, query, or control change after seeing
scores. INTERPRETATION: this calibrates the animal unit-discovery nulls —
success here with the SAME frozen matcher localises the animal failure to
the representation-on-animal-signal / matcher regime; failure here
constrains the matcher itself. No semantic claims (words are reference
LABELS for scoring, never meaning claims).

Usage:
    python scripts/librispeech_word_recovery.py --self-test
    python scripts/librispeech_word_recovery.py

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

from bengalese_candidate_ranking import kbest_candidates
from mfa_alignment_qa import LS_DIR, TG_DIR, _words, parse_textgrid
from natural_unit_reference_benchmark import _boot_rate, _iou

SR = 16000
WIN_MS, HOP_MS = 40.0, 10.0
SPECIES = "human"
MIN_FRAMES = 3
TRAIN_FRAC = 0.7
N_HIGH, N_MID, N_TOK = 15, 15, 5
MIN_EVAL_OCC = 10
K_GLOBAL = (5, 10)
BOOT_B = 1000
SEED = 0
CACHE = ROOT / "cache" / "libriword" / "trajs.npz"

_HOSTS: dict = {}


# ---------------------------------------------------------------- loading


def _load_corpus():
    """Exact-match utterances with word lists and speaker/split tags."""

    flacs = sorted(LS_DIR.rglob("*.flac"))
    trans = {}
    for tf in LS_DIR.rglob("*.trans.txt"):
        for line in tf.read_text().splitlines():
            uid, text = line.strip().split(" ", 1)
            trans[uid] = [t.lower() for t in text.split()]
    utts = []
    for f in flacs:
        uid = f.stem
        tg = TG_DIR / f.relative_to(LS_DIR).parent / f"{uid}.TextGrid"
        if not tg.exists():
            continue
        words = [(a, b, w) for a, b, w in _words(parse_textgrid(tg))]
        if [w.lower() for _, _, w in words] != trans.get(uid):
            continue  # tokenization-level mismatch — conservative exclusion
        spk = uid.split("-")[0]
        part = (
            "fit"
            if int(hashlib.sha1(spk.encode()).hexdigest()[:8], 16) % 100 < 100 * TRAIN_FRAC
            else "eval"
        )
        utts.append({"uid": uid, "spk": spk, "part": part, "words": words, "flac": str(f)})
    return utts


def _fingerprint_job(args):
    """Fingerprint trajectory (supported 12 features) for one utterance."""
    import numpy as np

    from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp
    from bioacoustic_pipeline.diagnostics.acoustic_derivatives import (
        _NAME_TO_IDX,
        supported_features,
    )
    from bioacoustic_pipeline.utils.audio import load_audio

    uid, flac, hop, win = args
    audio, sr = load_audio(flac, target_sr=SR)
    fidx = [_NAME_TO_IDX[n] for n in supported_features()]
    rows, i = [], 0
    while i * hop + win <= len(audio) and len(rows) < 4000:
        v = afp.extract(audio[i * hop : i * hop + win], sr, species=SPECIES).combined
        rows.append(np.asarray(v, dtype=np.float64)[fidx])
        i += 1
    if len(rows) < MIN_FRAMES:
        return uid, None, None
    tr = np.stack(rows)
    return uid, tr, ~np.all(tr == 0, axis=1)


def _build_cache(utts):
    """Content-cache all trajectories once (BLAKE-keyed by uid list)."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1("|".join(u["uid"] for u in utts).encode()).hexdigest()[:16]
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=False)
        if str(d["key"]) == key:
            return d
    hop, win = int(SR * HOP_MS / 1000), int(SR * WIN_MS / 1000)
    jobs = [(u["uid"], u["flac"], hop, win) for u in utts]
    with ProcessPoolExecutor(max_workers=24) as ex:
        results = list(ex.map(_fingerprint_job, jobs, chunksize=4))
    frames, offsets, names, valids = [], [0], [], []
    for uid, tr, valid in results:
        if tr is None:
            continue
        names.append(uid)
        frames.append(tr)
        valids.append(valid)
        offsets.append(offsets[-1] + len(tr))
    F = np.vstack(frames)
    V = np.concatenate(valids)
    np.savez_compressed(
        CACHE, key=key, frames=F, offsets=np.array(offsets), names=np.array(names), valid=V
    )
    return np.load(CACHE, allow_pickle=False)


def _word_frames(u, hop, win):
    """MFA word spans -> supported frame intervals (window-center-in-word)."""
    import numpy as np

    n_max = 4000  # fingerprint cap (see _fingerprint_job)
    centers = np.arange(n_max) * hop + win / 2.0
    out = []
    n_unsupported = 0
    for a, b, w in u["words"]:
        idx = np.where((centers >= a * SR) & (centers < b * SR))[0]
        idx = idx[idx < n_max]
        if len(idx) < MIN_FRAMES:
            n_unsupported += 1
            continue
        out.append(
            {
                "label": w.lower(),
                "s": int(idx[0]),
                "e": int(idx[-1]) + 1,
                "t0": a,
                "t1": b,
                "dur": b - a,
            }
        )
    return out, n_unsupported


# ----------------------------------------------------------------- search


def _query_search(args):
    """One label-blind query over all eval hosts -> pooled candidates."""
    q, q_uid = args
    out = []
    for uid, tr in _HOSTS.items():
        if uid == q_uid:
            continue
        cands, _ = kbest_candidates(q, tr, k=5)
        for c in cands:
            out.append((c["cost"], uid, c["interval"][0], c["interval"][1]))
    out.sort(key=lambda r: r[0])
    return out[: max(K_GLOBAL)]


def _score_query(cands, hosts_words, label):
    """Post-search scoring against word labels (IoU>=0.5 primary)."""

    res = {"n_cands": len(cands)}
    if not cands:
        return res
    rows = []
    for cost, uid, s, e in cands:
        ann = hosts_words[uid]
        best = None
        for a in ann:
            iou, ov = _iou(s, e, a["s"], a["e"])
            if ov > 0 and (best is None or iou > best[0]):
                best = (iou, a, ov)
        if best is None:
            rows.append(
                {
                    "correct": False,
                    "any": False,
                    "iou": 0.0,
                    "on": None,
                    "off": None,
                    "ctx": None,
                    "uid": uid,
                }
            )
            continue
        iou, a, ov = best
        same = a["label"] == label
        rows.append(
            {
                "correct": bool(same and iou >= 0.5),
                "any": bool(same and ov > 0),
                "iou": iou,
                "on": s - a["s"],  # frames; converted to ms at aggregation
                "off": e - a["e"],
                "ctx": (a["prev"], a["next"]),
                "uid": uid,
            }
        )
    res["top1"] = rows[0]
    res["rows"] = rows
    res["recall"] = {f"@{k}": any(r["correct"] for r in rows[:k]) for k in K_GLOBAL}
    res["recall_any"] = {f"@{k}": any(r["any"] for r in rows[:k]) for k in K_GLOBAL}
    return res


# ------------------------------------------------------------- self-test


def _self_test() -> None:
    """Synthetic two-'word' hosts with varied neighbours: the scoring and
    search-contract functions recover the word, localise it, and see
    context changes (mechanics check; real run exercises the pipeline)."""
    import numpy as np

    rng = np.random.RandomState(0)
    wA = np.sin(np.linspace(0, 6 * np.pi, 9))[:, None] * np.ones((1, 12))
    wB = np.cos(np.linspace(0, 4 * np.pi, 9))[:, None] * np.ones((1, 12)) * 0.7
    hosts, hosts_words = {}, {}
    labels_cycle = ["x", "y", "z"]
    for h in range(6):
        ann, chunks, pos = [], [], 0
        for j in range(rng.randint(4, 7)):
            which = "A" if rng.rand() < 0.5 else "B"
            tmpl = wA if which == "A" else wB
            chunks.append(0.3 * rng.randn(5, 12))
            chunks.append(tmpl + 0.02 * rng.randn(9, 12))
            ann.append(
                {
                    "label": which,
                    "s": pos + 5,
                    "e": pos + 14,
                    "prev": labels_cycle[j % 3],
                    "next": labels_cycle[(j + 1) % 3],
                }
            )
            pos += 14
        hosts[f"h{h}"] = np.vstack(chunks)
        hosts_words[f"h{h}"] = ann
    q = hosts["h0"][hosts_words["h0"][0]["s"] : hosts_words["h0"][0]["e"]]
    global _HOSTS
    _HOSTS = hosts
    cands = _query_search((q, "h0"))
    res = _score_query(cands, hosts_words, hosts_words["h0"][0]["label"])
    assert res["top1"]["correct"], res["top1"]
    assert res["recall"]["@5"], res
    ious = [r["iou"] for r in res["rows"] if r["correct"]]
    assert float(np.median(ious)) >= 0.5, ious
    ctx_changes = sum(
        1
        for r in res["rows"][:10]
        if r["correct"] and r["ctx"] != (hosts_words["h0"][0]["prev"], hosts_words["h0"][0]["next"])
    )
    assert ctx_changes >= 1, res["rows"][:5]
    print(
        f"SELF-TEST OK (synthetic word: top-1 correct, IoU {np.median(ious):.2f}, "
        f"context changes {ctx_changes})",
        flush=True,
    )


# ------------------------------------------------------------------ main


def _real_run() -> None:
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    hop, win = int(SR * HOP_MS / 1000), int(SR * WIN_MS / 1000)
    utts = _load_corpus()
    fit = [u for u in utts if u["part"] == "fit"]
    ev = [u for u in utts if u["part"] == "eval"]
    print(f"exact-match utterances: {len(utts)} (fit {len(fit)} / eval {len(ev)})", flush=True)

    d = _build_cache(utts)
    names = list(d["names"])
    off, F, V = d["offsets"], d["frames"], d["valid"]
    traj = {n: (F[off[i] : off[i + 1]], V[off[i] : off[i + 1]]) for i, n in enumerate(names)}

    # supported word inventory + contexts (labels: from here on they shape
    # query SELECTION; the search itself remains label-blind)
    for u in utts:
        ann, n_unsup = _word_frames(u, hop, win)
        u["ann"] = ann
        u["n_unsupported"] = n_unsup
        for i, a in enumerate(ann):
            a["prev"] = ann[i - 1]["label"] if i > 0 else None
            a["next"] = ann[i + 1]["label"] if i + 1 < len(ann) else None

    eval_occ, fit_occ, eval_ctx, eval_dur = {}, {}, {}, {}
    eval_tokens_total = 0
    for u in ev:
        eval_tokens_total += len(u["ann"])
        for a in u["ann"]:
            eval_occ[a["label"]] = eval_occ.get(a["label"], 0) + 1
            eval_ctx.setdefault(a["label"], []).append((a["prev"], a["next"]))
            eval_dur.setdefault(a["label"], []).append(a["dur"])
    for u in fit:
        for a in u["ann"]:
            fit_occ.setdefault(a["label"], []).append((u, a))

    eligible = sorted([w for w, n in eval_occ.items() if n >= MIN_EVAL_OCC])
    rng = np.random.RandomState(SEED)
    high = sorted(eligible, key=lambda w: -eval_occ[w])[: min(N_HIGH, len(eligible))]
    mid_pool = [w for w in eligible if MIN_EVAL_OCC <= eval_occ[w] <= 49 and w not in high]
    mid = sorted(rng.choice(mid_pool, size=min(N_MID, len(mid_pool)), replace=False).tolist())
    types = high + mid
    print(f"word types selected: {len(types)} (high {len(high)} / mid {len(mid)})", flush=True)

    # matched controls: median duration +-25%, log eval freq +-0.5
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

    def _pick_tokens(w, n):
        """n fit tokens nearest the type's median fit duration."""
        toks = fit_occ.get(w, [])
        if not toks:
            return []
        durs = np.array([a["dur"] for _, a in toks])
        med = float(np.median(durs))
        order = np.argsort(np.abs(durs - med))
        return [toks[i] for i in order[:n]]

    hosts_words = {u["uid"]: u["ann"] for u in ev}
    global _HOSTS
    _HOSTS = {
        u["uid"]: traj[u["uid"]][0]
        for u in ev
        if u["uid"] in traj and traj[u["uid"]][0].shape[0] >= 30
    }

    def _run_arm(queries):
        """queries: list of (word_label, token_u, token_a) -> scored results."""
        jobs, meta = [], []
        for w, u, a in queries:
            q = traj[u["uid"]][0][a["s"] : a["e"]]
            if len(q) < MIN_FRAMES:
                continue
            jobs.append((q, u["uid"]))
            meta.append((w, u, a))
        with ProcessPoolExecutor(max_workers=24) as ex:
            all_cands = list(ex.map(_query_search, jobs, chunksize=1))
        return [(w, u, a, _score_query(c, hosts_words, w)) for (w, u, a), c in zip(meta, all_cands)]

    queries, ctrl_queries = [], []
    for w in types:
        for u, a in _pick_tokens(w, N_TOK):
            queries.append((w, u, a))
        c = controls[w]
        if c:
            for u, a in _pick_tokens(c, N_TOK):
                ctrl_queries.append((c, u, a))
    print(
        f"cross-speaker queries: {len(queries)} + {len(ctrl_queries)} matched-control", flush=True
    )
    scored = _run_arm(queries)
    ctrl_scored = _run_arm(ctrl_queries) if ctrl_queries else []

    # same-speaker arm: query from eval speaker, hosts = same speaker only
    same_scored = []
    spk_utts = {}
    for u in ev:
        spk_utts.setdefault(u["spk"], []).append(u)
    for w in types:
        n_speakers = 0
        for spk, us in sorted(spk_utts.items()):
            if n_speakers >= 3:
                break
            hosts_spk = [x for x in us if x["uid"] in _HOSTS]
            if len(hosts_spk) < 2:
                continue
            src = next((x for x in hosts_spk for a in x["ann"] if a["label"] == w), None)
            if src is None:
                continue
            a = next(a for a in src["ann"] if a["label"] == w)
            q = traj[src["uid"]][0][a["s"] : a["e"]]
            if len(q) < MIN_FRAMES:
                continue
            _HOSTS_local = {
                x["uid"]: traj[x["uid"]][0] for x in hosts_spk if x["uid"] != src["uid"]
            }
            cands = []
            for uid, tr in _HOSTS_local.items():
                for c in kbest_candidates(q, tr, k=5)[0]:
                    cands.append((c["cost"], uid, c["interval"][0], c["interval"][1]))
            cands.sort(key=lambda r: r[0])
            same_scored.append((w, src, a, _score_query(cands[: max(K_GLOBAL)], hosts_words, w)))
            n_speakers += 1

    # ---- aggregate
    def _arm_stats(rows, key_top="top1"):
        import numpy as np

        per_tok_r1 = [(1 if r[key_top]["correct"] else 0, 1) for _, _, _, r in rows if key_top in r]
        per_tok_r10 = [(1 if r["recall"]["@10"] else 0, 1) for _, _, _, r in rows if "recall" in r]
        ious = [r[key_top]["iou"] for _, _, _, r in rows if key_top in r and r[key_top]["correct"]]
        ons = [
            abs(r[key_top]["on"]) for _, _, _, r in rows if key_top in r and r[key_top]["correct"]
        ]
        offs = [
            abs(r[key_top]["off"]) for _, _, _, r in rows if key_top in r and r[key_top]["correct"]
        ]
        miss = sum(1 for _, _, _, r in rows if key_top in r and not r[key_top]["any"])
        return {
            "n_queries": len(rows),
            "recall1": _boot_rate(per_tok_r1, b=BOOT_B),
            "recall10": _boot_rate(per_tok_r10, b=BOOT_B),
            "median_iou_correct_top1": round(float(np.median(ious)), 4) if ious else None,
            "median_onset_err_ms": round(float(np.median(ons)) * 10.0, 1) if ons else None,
            "median_offset_err_ms": round(float(np.median(offs)) * 10.0, 1) if offs else None,
            "complete_miss_fraction": round(miss / max(len(rows), 1), 4),
        }

    # contextual reuse: among correct top-10 candidates
    ctx_rows = []
    for w, u, a, r in scored:
        if "rows" not in r:
            continue
        qctx = (a["prev"], a["next"])
        corr = [c for c in r["rows"][:10] if c["correct"]]
        if not corr:
            continue
        change = sum(1 for c in corr if c["ctx"] != qctx) / len(corr)
        ctxs = eval_ctx.get(w, [])
        ref = 1 - sum((ctxs.count(c) / len(ctxs)) ** 2 for c in set(ctxs)) if ctxs else None
        ctx_rows.append(
            {
                "label": w,
                "n_correct": len(corr),
                "change": round(change, 4),
                "ref": round(ref, 4) if ref is not None else None,
                "dev": round(change - ref, 4) if ref is not None else None,
            }
        )

    avail = {}
    for w in types:
        rates = [
            sum(1 for a in u["ann"] if a["label"] == w) / len(u["ann"]) for u in ev if u["ann"]
        ]
        avail[w] = round(float(np.mean(rates)), 6) if rates else None

    cross = _arm_stats(scored)
    ctrl = _arm_stats(ctrl_scored) if ctrl_scored else None
    same = _arm_stats(same_scored) if same_scored else None

    def _tier(cross, ctrl):
        if ctrl and ctrl["recall1"] and cross["recall1"]:
            if cross["recall1"]["point"] >= 0.2 and cross["recall1"]["point"] >= 2 * max(
                ctrl["recall1"]["point"], 1e-9
            ):
                return "recovery demonstrated"
            if cross["recall1"]["point"] < 0.05 or cross["recall1"]["point"] < 2 * max(
                ctrl["recall1"]["point"], 1e-9
            ):
                return "absent"
        return "indeterminate"

    report = {
        "preregistration": __doc__.split("PREREGISTRATION")[1].split("MATERIAL")[0].strip(),
        "reference_provenance": "MFA public alignments (Zenodo 2619474, "
        "CC-BY-4.0); MACHINE alignments — human validation PENDING "
        "(outputs/mfa_alignment_review/); results carry 'MFA-reference, "
        "human-validation pending'",
        "labels_hidden_during_search": True,
        "corpus": {
            "exact_match_utterances": len(utts),
            "fit": len(fit),
            "eval": len(ev),
            "eval_hosts_fingerprinted": len(_HOSTS),
            "unsupported_word_tokens": int(sum(u.get("n_unsupported", 0) for u in utts)),
            "word_types": len(types),
        },
        "cross_speaker": cross,
        "matched_control": ctrl,
        "same_speaker": same,
        "availability_floor": avail,
        "contextual_reuse": ctx_rows,
        "tiers": {
            "recovery": _tier(cross, ctrl),
            "localisation": "adequate"
            if cross["median_iou_correct_top1"] is not None
            and cross["median_iou_correct_top1"] >= 0.5
            else "not adequate",
        },
        "stop_rule": "one run; no parameter or query change after scores",
        "interpretation": "calibrates the animal unit-discovery nulls; no "
        "semantic claims — word labels are scoring references only",
    }
    out = ROOT / "outputs" / "librispeech_word_recovery.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {k: report[k] for k in ("cross_speaker", "matched_control", "same_speaker", "tiers")},
            indent=1,
        ),
        flush=True,
    )
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    _real_run()


if __name__ == "__main__":
    main()
