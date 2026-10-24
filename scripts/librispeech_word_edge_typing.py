#!/usr/bin/env python3
"""LibriSpeech word-edge typing benchmark — calibrating boundary-SELECTION
logic where the unit hierarchy is known. Advisor-directed 2026-10-20.

FRAMING: LibriSpeech calibrates the ARCHITECTURE (typed selection,
decoder structure, evaluation discipline) — never a human boundary model
transplanted to animals. The hierarchy utterance->word->phoneme teaches
the distinction this pipeline lacks: "real acoustic change" is not "unit
boundary" (inside one word live many strong phonetic changes). Firewall:
human CPC context is permitted ON HUMANS (in-house human model); NOTHING
learned here (weights, thresholds, duration priors) transfers to animal
data.

LADDER (frozen; trained on fit speakers, scored prospectively on eval
speakers):
  L1 raw high-recall NBD proposals (regime fixed from the Bengalese
     study: floors 0.02/0.015, fire-on-drop 0.015, debounce 20 ms) —
     proposal ceiling; REQUIREMENT R_proposal >= 0.9 (prospective floor,
     checked not tuned);
  L2 scalar typed selector: logistic 4-way {word-onset, word-offset,
     internal-phoneme, gap} on the standardised NBD proposal features
     (same FEATURES as the Bengalese study); kept = onset+offset;
  L3 + human CPC context: pre/post/diff CPC frame embeddings around each
     proposal via the NBD calibrator's proposal_context helper
     (guard: CPCBackbone.available; arm reports unavailable otherwise);
  L4 best selector + constrained decoder/abstain: onset->offset pairing
     with word duration in [0.10, 2.0] s, unmatched edges dropped.

LABELS (MFA reference; carries 'MFA-reference, human-validation
pending'): word-onset / word-offset within tol of word span edges;
internal-phoneme = inside a word span, >tol from its edges; gap = outside
all word spans. PRIMARY ENDPOINT: F1_word-boundary (onset+offset union,
tol 3 frames = 30 ms) — NOT combined word+phoneme. Separate: onset /
offset recall, over-/under-segmentation, segment IoU, word-level
one-to-one recall. CALIBRATION: P(type | proposal) raw; P(word-edge |
selected) per stage; delta_edge-purity.

DECISION RULES (fixed now): "L2 useful" if F1_word-boundary improves on
the raw purity-implied F1; "CPC adds" if L3 F1 > L2 F1 + 0.05; "decoder
adds" if L4 word-one-to-one recall >= best selector's + 0.05; proposal
floor checked. One run; nothing changed after eval scores.

Usage:
    python scripts/librispeech_word_edge_typing.py --self-test
    python scripts/librispeech_word_edge_typing.py [--n-fit 800] [--n-eval 400]

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

from bengalese_boundary_decomposition import TOL_PRIMARY, _match_boundaries
from bengalese_proposal_selection import FEATURES, REGIMES
from librispeech_word_recovery import SR, _load_corpus
from mfa_alignment_qa import LS_DIR, TG_DIR, parse_textgrid

TOL = TOL_PRIMARY
HOP_MS = 10.0
WIN_MS = 40.0
MIN_WORD_S, MAX_WORD_S = 0.10, 2.0
SEED = 0


def _proposals_flac(flac):
    """High-recall NBD proposals + feature matrix for one flac utterance."""
    import numpy as np

    from bioacoustic_pipeline.segmentation import nbd_proposal as nbd
    from bioacoustic_pipeline.utils.audio import load_audio

    audio, sr = load_audio(flac, target_sr=SR)
    if audio is None or len(audio) < SR // 2:
        return None
    sig = nbd.compute_signals(audio, SR, hop_ms=HOP_MS)
    props = nbd.generate_proposals(sig, **REGIMES["A2_highrecall"])
    X = [
        [
            p.energy_change,
            p.spectral_flux,
            p.band_flux,
            np.log1p(p.local_rms),
            p.rms_slope,
            p.duration_since_prev_ms,
            p.rms_ratio,
            p.flux_contrast,
            p.energy_contrast,
            p.flux_asym,
            p.duration_to_next_ms,
            p.spectral_drop,
            float("energy" in p.detector),
            float("spectral" in p.detector),
            float("spectral_drop" in p.detector),
        ]
        for p in props
    ]
    return {
        "props": props,
        "X": np.array(X) if X else np.zeros((0, len(FEATURES))),
        "frames": [p.frame for p in props],
        "n_frames": sig.n_frames,
        "duration_s": len(audio) / SR,
    }


def _word_frames(u, hop, win):
    """MFA word spans -> (start, end) frame intervals, window-center rule."""
    import numpy as np

    n_max = 4000
    centers = np.arange(n_max) * hop + win / 2.0
    out = []
    for a, b, _w in u["words"]:
        idx = np.where((centers >= a * SR) & (centers < b * SR))[0]
        idx = idx[idx < n_max]
        if len(idx) >= 1:
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def _label_frame(frame, word_frames):
    for s, e in word_frames:
        if abs(frame - s) <= TOL:
            return "onset"
        if abs(frame - (e - 1)) <= TOL:
            return "offset"
    for s, e in word_frames:
        if s <= frame < e:
            return "internal"
    return "gap"


def _decode(typed_preds):
    """Constrained decoder: pair onsets with following offsets within
    [MIN_WORD_S, MAX_WORD_S]; drop unmatched edges (abstention)."""
    out, open_onset = [], None
    for f, t in sorted(typed_preds.items()):
        if t == "onset":
            open_onset = f if open_onset is None else open_onset
        elif t == "offset" and open_onset is not None:
            dur_s = (f - open_onset) * HOP_MS / 1000.0
            if MIN_WORD_S <= dur_s <= MAX_WORD_S:
                out.extend([open_onset, f])
            open_onset = None
    return out


def _eval_kept(kept, word_frames, n_frames):
    onsets = [s for s, _ in word_frames]
    offsets = [e - 1 for _, e in word_frames]
    true_union = sorted(set(onsets) | set(offsets))
    pairs, _, _ = _match_boundaries(true_union, kept, TOL)
    res = {
        "n_kept": len(kept),
        "n_true": len(true_union),
        "n_matched": len(pairs),
        "onset_matched": sum(1 for o in onsets if any(abs(o - k) <= TOL for k in kept)),
        "onset_total": len(onsets),
        "offset_matched": sum(1 for o in offsets if any(abs(o - k) <= TOL for k in kept)),
        "offset_total": len(offsets),
        "seg_ious": [],
        "word_one_to_one": 0,
        "over_seg": 0,
        "under_seg": 0,
        "n_words": len(word_frames),
    }
    b = [0, *sorted(kept), n_frames]
    used = set()
    for i in range(len(b) - 1):
        s, e = b[i], b[i + 1]
        best = None
        for j, (ws, we) in enumerate(word_frames):
            ov = min(e, we) - max(s, ws)
            if ov <= 0:
                continue
            iou = ov / (max(e, we) - min(s, ws))
            if best is None or iou > best[0]:
                best = (iou, j)
        if best:
            res["seg_ious"].append(best[0])
            if best[0] >= 0.5 and best[1] not in used:
                used.add(best[1])
                res["word_one_to_one"] += 1
    res["over_seg"] = sum(
        1 for ws, we in word_frames if sum(1 for k in kept if ws + TOL < k < we - 1 - TOL) > 0
    )
    res["under_seg"] = sum(
        1 for ws, we in word_frames if not any(ws + TOL <= k <= we - 1 - TOL for k in kept)
    )
    return res


def _agg(rows):
    import numpy as np

    n_kept = sum(r["n_kept"] for r in rows)
    n_true = sum(r["n_true"] for r in rows)
    n_m = sum(r["n_matched"] for r in rows)
    P = n_m / max(n_kept, 1)
    R = n_m / max(n_true, 1)
    ious = [i for r in rows for i in r["seg_ious"]]
    n_words = sum(r["n_words"] for r in rows)
    return {
        "f1_word_boundary": round(2 * P * R / max(P + R, 1e-9), 4),
        "boundary_precision": round(P, 4),
        "boundary_recall": round(R, 4),
        "onset_recall": round(
            sum(r["onset_matched"] for r in rows) / max(sum(r["onset_total"] for r in rows), 1), 4
        ),
        "offset_recall": round(
            sum(r["offset_matched"] for r in rows) / max(sum(r["offset_total"] for r in rows), 1), 4
        ),
        "overseg_rate": round(sum(r["over_seg"] for r in rows) / max(n_words, 1), 4),
        "underseg_rate": round(sum(r["under_seg"] for r in rows) / max(n_words, 1), 4),
        "median_segment_iou": round(float(np.median(ious)), 4) if ious else None,
        "word_one_to_one_recall": round(
            sum(r["word_one_to_one"] for r in rows) / max(n_words, 1), 4
        ),
        "n_kept": n_kept,
    }


def _self_test() -> None:
    word_frames = [(10, 40), (60, 110), (150, 190)]  # 290/490/390 ms words
    frames = [10, 12, 20, 39, 60, 65, 109, 150, 189, 250]
    labs = [_label_frame(f, word_frames) for f in frames]
    assert labs == [
        "onset",
        "onset",
        "internal",
        "offset",
        "onset",
        "internal",
        "offset",
        "onset",
        "offset",
        "gap",
    ], labs
    preds = {
        10: "onset",
        20: "internal",
        39: "offset",
        60: "onset",
        65: "internal",
        109: "offset",
        150: "onset",
        189: "offset",
        250: "offset",
    }
    kept = _decode(preds)
    assert kept == [10, 39, 60, 109, 150, 189], kept  # unmatched 250 abstained
    r = _eval_kept(kept, word_frames, 300)
    assert r["word_one_to_one"] == 3 and r["n_kept"] == 6, r
    print("SELF-TEST OK (4-way labels; decoder pairs + abstains; metrics)", flush=True)


def _prep(args):
    """Module-level (picklable): proposals + word frames for one utterance."""
    u, hop, win = args
    pr = _proposals_flac(u["flac"])
    if pr is None:
        return None
    pr["word_frames"] = _word_frames(u, hop, win)
    return pr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-fit", type=int, default=800)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return

    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    hop, win = int(SR * HOP_MS / 1000), int(SR * WIN_MS / 1000)
    utts = _load_corpus()
    fit = [u for u in utts if u["part"] == "fit"][: args.n_fit]
    ev = [u for u in utts if u["part"] == "eval"][: args.n_eval]
    print(f"utterances: fit {len(fit)} / eval {len(ev)} (speaker-disjoint)", flush=True)
    for u in fit + ev:
        tg = (
            TG_DIR / Path(u["flac"]).relative_to(LS_DIR).parent / f"{Path(u['flac']).stem}.TextGrid"
        )
        tiers = parse_textgrid(tg)
        u["words"] = [(a, b, w) for a, b, w in tiers.get("words", []) if w.strip()]

    with ProcessPoolExecutor(max_workers=24) as ex:
        # keep utterances aligned with their prepared proposals (drop pairs)
        pairs_fit = [
            (r, u)
            for r, u in zip(ex.map(_prep, [(u, hop, win) for u in fit], chunksize=4), fit)
            if r
        ]
        pairs_ev = [
            (r, u) for r, u in zip(ex.map(_prep, [(u, hop, win) for u in ev], chunksize=4), ev) if r
        ]
    prep_fit, fit_al = [p[0] for p in pairs_fit], [p[1] for p in pairs_fit]
    prep_ev, ev_al = [p[0] for p in pairs_ev], [p[1] for p in pairs_ev]
    print(f"proposals prepared: fit {len(prep_fit)} / eval {len(prep_ev)}", flush=True)

    # ---- L1 raw ceiling + proposal composition
    def _raw(prep):
        n_m = n_true = n_prop = 0
        comp = {"onset": 0, "offset": 0, "internal": 0, "gap": 0}
        for pr in prep:
            wf = pr["word_frames"]
            true_union = sorted({s for s, _ in wf} | {e - 1 for _, e in wf})
            n_true += len(true_union)
            n_prop += len(pr["frames"])
            pairs, _, _ = _match_boundaries(true_union, pr["frames"], TOL)
            n_m += len(pairs)
            for f in pr["frames"]:
                comp[_label_frame(f, wf)] += 1
        R = n_m / max(n_true, 1)
        return {
            "proposal_recall": round(R, 4),
            "n_proposals": n_prop,
            "P_type_given_proposal": {k: round(v / max(n_prop, 1), 4) for k, v in comp.items()},
            "raw_edge_purity": round((comp["onset"] + comp["offset"]) / max(n_prop, 1), 4),
        }

    l1_ev = _raw(prep_ev)
    print(
        f"L1 eval: recall {l1_ev['proposal_recall']} edge purity {l1_ev['raw_edge_purity']}",
        flush=True,
    )

    # ---- CPC context (human-only; guarded)
    cpc = None
    try:
        from bioacoustic_pipeline.embeddings.backbones import CPCBackbone

        bb = CPCBackbone()
        if getattr(bb, "available", False):
            cpc = bb
    except Exception as e:  # noqa: BLE001
        print(f"CPC unavailable ({type(e).__name__}: {e}); L3 reports unavailable", flush=True)

    def _cpc_context(prep, utts_al):
        from bioacoustic_pipeline.segmentation.nbd_calibrator import proposal_context
        from bioacoustic_pipeline.utils.audio import load_audio

        feats = []
        for pr, u in zip(prep, utts_al):
            audio, _sr = load_audio(u["flac"], target_sr=SR)
            emb = cpc.embed_frames(audio)
            feats.append(proposal_context(pr["props"], np.asarray(emb)))
        return np.vstack(feats) if feats else None

    Xd = np.vstack([pr["X"] for pr in prep_fit])
    yd = [_label_frame(f, pr["word_frames"]) for pr in prep_fit for f in pr["frames"]]
    scaler = StandardScaler().fit(Xd)

    def _stage(prep, model, ctx=None, decode=False, scaler2=None):
        rows, sel_edge, sel_n = [], 0, 0
        off = 0
        for pr in prep:
            X = scaler.transform(pr["X"])
            if ctx is not None:
                n = len(pr["frames"])
                X = np.hstack([X, ctx[off : off + n]])
                off += n
            if scaler2 is not None:
                X = scaler2.transform(X)
            pred = model.predict(X)
            typed_preds = dict(zip(pr["frames"], pred))
            kept = [f for f, p in typed_preds.items() if p in ("onset", "offset")]
            if decode:
                kept = _decode(typed_preds)
            rows.append(_eval_kept(kept, pr["word_frames"], pr["n_frames"]))
            for f in kept:
                sel_edge += _label_frame(f, pr["word_frames"]) in ("onset", "offset")
                sel_n += 1
        out = _agg(rows)
        out["selected_edge_purity"] = round(sel_edge / max(sel_n, 1), 4)
        out["delta_edge_purity"] = round(out["selected_edge_purity"] - l1_ev["raw_edge_purity"], 4)
        return out

    # L2 scalar typed
    typed2 = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(
        scaler.transform(Xd), yd
    )
    l2 = _stage(prep_ev, typed2)
    print(
        f"L2 eval: F1 {l2['f1_word_boundary']} edge purity {l2['selected_edge_purity']}", flush=True
    )

    # L3 + CPC context
    l3 = None
    if cpc is not None:
        Cf = _cpc_context(prep_fit, fit_al)
        Ce = _cpc_context(prep_ev, ev_al)
        if Cf is not None and Cf.size and Ce is not None and Ce.size:
            Xd3 = np.hstack([scaler.transform(Xd), Cf])
            sc3 = StandardScaler().fit(Xd3)
            typed3 = LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=SEED
            ).fit(sc3.transform(Xd3), yd)
            l3 = _stage(prep_ev, typed3, ctx=Ce, scaler2=sc3)
            print(
                f"L3 eval: F1 {l3['f1_word_boundary']} edge purity {l3['selected_edge_purity']}",
                flush=True,
            )
        else:
            print("CPC context empty — L3 unavailable", flush=True)

    # L4 decoder on the best selector
    if l3 is not None and l3["f1_word_boundary"] >= l2["f1_word_boundary"]:
        l4 = _stage(prep_ev, typed3, ctx=Ce, scaler2=sc3, decode=True)
    else:
        l4 = _stage(prep_ev, typed2, decode=True)
    print(
        f"L4 eval: F1 {l4['f1_word_boundary']} word 1-1 {l4['word_one_to_one_recall']}", flush=True
    )

    raw_f1_ref = (2 * l1_ev["raw_edge_purity"] * l1_ev["proposal_recall"]) / max(
        l1_ev["raw_edge_purity"] + l1_ev["proposal_recall"], 1e-9
    )
    verdicts = {
        "proposal_floor_met": bool(l1_ev["proposal_recall"] >= 0.9),
        "l2_useful": bool(l2["f1_word_boundary"] > round(raw_f1_ref, 4)),
        "cpc_adds": (bool(l3["f1_word_boundary"] > l2["f1_word_boundary"] + 0.05) if l3 else None),
        "decoder_adds": bool(
            l4["word_one_to_one_recall"] >= (l3 or l2)["word_one_to_one_recall"] + 0.05
        ),
    }

    report = {
        "study": "LibriSpeech word-edge typing benchmark (architecture "
        "calibration; only architecture/metrics/discipline transfer — "
        "never human weights, thresholds, or duration priors)",
        "reference_provenance": "MFA machine alignments — human-validation "
        "pending (outputs/mfa_alignment_review/)",
        "ladder": {
            "L1_raw_highrecall": l1_ev,
            "L2_scalar_typed": l2,
            "L3_cpc_context": l3 if l3 is not None else "CPC unavailable",
            "L4_decoder": l4,
            "raw_purity_implied_f1_reference": round(float(raw_f1_ref), 4),
        },
        "verdicts": verdicts,
        "stop_rule": "one run; nothing changed after eval scores",
    }
    out = ROOT / "outputs" / "librispeech_word_edge_typing.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(verdicts, indent=1), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
