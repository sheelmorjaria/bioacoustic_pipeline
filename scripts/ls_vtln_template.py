#!/usr/bin/env python3
"""VTLN (Vocal Tract Length Normalization) + fingerprint DBA consensus template.

§27 proved that DBA averaging fails on the raw fingerprint (Δ −0.040) because the
speaker/word entanglement is non-additive — the primary physical driver is Vocal Tract
Length (VTL), which causes non-linear formant scaling across speakers. VTLN warps the
frequency axis (via audio resampling by factor α) to a canonical formant space BEFORE
fingerprint extraction, attacking the mechanism at its source rather than algebraically
after the fact.

Experiment: for each word instance, grid-search α ∈ [0.88, 1.12] (step 0.04); resample
the audio by α, extract the fp-window trajectory, pick the α that minimises the average
DTW distance to the word's other-speaker instances. This normalises every instance into a
common formant space. Then run the DBA consensus template + retrieval (same §27 protocol).

If the VTLN-normalised DBA template beats raw cross-speaker NN, VTLN has successfully
removed the physical speaker entanglement that defeated algebraic centering/averaging —
and it's deterministic, physics-grounded, species-agnostic (no learned weights, no firewall
issue). If it fails, the entanglement is not solely VTL-driven.

Re-reads the manifest's audio (stores fp trajectories, not raw audio). Human-speech word
GT honest; semantic_validation stays none.

Run:  python scripts/ls_vtln_template.py

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import glob
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402
from bioacoustic_pipeline.utils.audio import load_audio  # noqa: E402
from ls_consensus_template import dba, dtw_cost  # noqa: E402

LS = Path("/home/sheel/birdsong_analysis/data/LibriSpeech/train-clean-360")
MANIFEST = ROOT / "outputs" / "ls_unified" / "manifest.json"
OUT = ROOT / "outputs" / "ls_unified" / "vtln_recs.pkl"
TOP_WORDS = 25
EVAL_CAP = 250
MIN_SPEAKERS = 3
ALPHAS = [0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12]
HOP = 0.03
WIN = 0.04


def _resample(audio, sr, alpha):
    """VTLN frequency warp via time-domain resampling (freq scales by 1/alpha)."""
    from scipy.signal import resample_poly  # noqa: PLC0415

    n = max(1, int(len(audio) / alpha))
    g = np.gcd(n, len(audio))
    return resample_poly(audio, n // g, len(audio) // g).astype(np.float32)


def _fp_trajectory(audio, sr):
    """fp-window trajectory (list of 43-D) from audio, matching ls_unified's method."""
    fp_fn = lambda seg, s: afp.extract(seg, s, species="human").combined  # noqa: E731
    hop_s, win_s = int(HOP * sr), int(WIN * sr)
    fps = []
    for s in range(0, len(audio) - win_s, hop_s):
        seg = audio[s : s + win_s]
        if len(seg) >= win_s:
            v = np.asarray(fp_fn(seg, sr), dtype=np.float32)
            if np.any(v):
                fps.append(v)
    return fps


def main():
    m = json.loads(MANIFEST.read_text())
    file_ids = m["file_ids"]
    word_spans = m["word_spans"]
    n_words = m["n_words"]
    print(f"manifest: {m['n_files']} files, {n_words} words", flush=True)

    # resolve paths
    all_flacs = glob.glob(str(LS / "**" / "*.flac"), recursive=True)
    base2path = {Path(f).name: f for f in all_flacs}
    by_file = defaultdict(list)
    for wi, span in enumerate(word_spans):
        by_file.setdefault(span[0], []).append(wi)
    print(f"  {len(by_file)} files to process | {len(ALPHAS)} alpha values", flush=True)

    # pass 1: extract raw fp trajectories (alpha=1.0) for ALL words (reference for grid-search)
    print("pass 1: extracting alpha=1.0 trajectories...", flush=True)
    raw_fps = [None] * n_words
    for file_idx in sorted(by_file):
        fid = file_ids[file_idx]
        path = base2path.get(fid)
        if path is None:
            continue
        audio, sr = load_audio(path, target_sr=16000)
        if audio is None:
            continue
        for wi in by_file[file_idx]:
            ws_ms, we_ms = word_spans[wi][1], word_spans[wi][2]
            si, ei = int(ws_ms / 1000 * sr), int(we_ms / 1000 * sr)
            seg = audio[si:ei]
            if len(seg) >= int(0.08 * sr):
                raw_fps[wi] = _fp_trajectory(seg, sr)
        if file_idx % 50 == 0:
            n_have = sum(1 for x in raw_fps if x)
            print(f"  {file_idx}/{len(by_file)} files → {n_have} words", flush=True)

    # build word → instances index for grid-search reference
    by_word = defaultdict(list)
    by_word_spk = defaultdict(set)
    for wi in range(n_words):
        if raw_fps[wi] and len(raw_fps[wi]) >= 2:
            w = word_spans[wi][3]
            by_word[w].append(wi)
            by_word_spk[w].add(word_spans[wi][4])
    cand = sorted(
        (w for w in by_word if len(by_word_spk[w]) >= MIN_SPEAKERS),
        key=lambda w: -len(by_word[w]),
    )
    words = cand[:TOP_WORDS]
    eval_pool = [wi for w in words for wi in by_word[w]]
    print(f"  {len(words)} multi-speaker words | {len(eval_pool)} instances in scope", flush=True)

    # pass 2: VTLN grid-search — for each instance, find alpha that minimises avg DTW to
    # other-speaker instances of the same word (at alpha=1.0)
    print("pass 2: VTLN grid-search...", flush=True)
    vtln_fps = [None] * n_words
    vtln_alpha = [None] * n_words
    # cache: load audio per file once for the resampling
    file_audio_cache = {}

    for wi in eval_pool:
        w = word_spans[wi][3]
        fi = word_spans[wi][0]
        fid = file_ids[fi]
        path = base2path.get(fid)
        if path is None:
            continue
        if fid not in file_audio_cache:
            file_audio_cache[fid], _ = load_audio(path, target_sr=16000)
        audio = file_audio_cache[fid]
        if audio is None:
            continue
        sr = 16000
        ws_ms, we_ms = word_spans[wi][1], word_spans[wi][2]
        si, ei = int(ws_ms / 1000 * sr), int(we_ms / 1000 * sr)
        seg = audio[si:ei]
        if len(seg) < int(0.08 * sr):
            continue
        # reference: other-speaker instances' raw fps (alpha=1.0)
        ref_fps = [
            raw_fps[j]
            for j in by_word[w]
            if j != wi and word_spans[j][4] != word_spans[wi][4] and raw_fps[j]
        ]
        if not ref_fps:
            continue
        best_alpha, best_dist, best_fps = 1.0, float("inf"), raw_fps[wi]
        for alpha in ALPHAS:
            warped = _resample(seg, sr, alpha)
            fps = _fp_trajectory(warped, sr)
            if len(fps) < 2:
                continue
            dist = np.mean([dtw_cost(fps, r) for r in ref_fps[:5]])  # cap ref for speed
            if dist < best_dist:
                best_alpha, best_dist, best_fps = alpha, dist, fps
        vtln_fps[wi] = best_fps
        vtln_alpha[wi] = best_alpha

    n_have = sum(1 for x in vtln_fps if x)
    print(f"  VTLN done: {n_have}/{len(eval_pool)} instances normalised", flush=True)

    # save
    with OUT.open("wb") as f:
        pickle.dump(vtln_fps, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved {OUT}", flush=True)

    # DBA consensus template + retrieval (same §27 protocol)
    print(f"\n{'=' * 80}\nVTLN-NORMALISED DBA CONSENSUS TEMPLATE vs RAW NN\n{'=' * 80}")
    rng = random.Random(0)
    # build recs-like structure for the DBA/retrieval functions
    recs_v = [
        {
            "fp": vtln_fps[wi] if vtln_fps[wi] else [],
            "word": word_spans[wi][3],
            "speaker": word_spans[wi][4],
        }
        for wi in range(n_words)
    ]
    # restrict to eval_pool (in-scope instances)
    scope = [wi for wi in eval_pool if vtln_fps[wi]]
    fp_seq = {
        "fp": [recs_v[wi]["fp"] for wi in scope],
        "word": [recs_v[wi]["word"] for wi in scope],
        "speaker": [recs_v[wi]["speaker"] for wi in scope],
    }
    # rebuild by_word over scope
    by_word_s = defaultdict(list)
    for j, wi in enumerate(scope):
        by_word_s[fp_seq["word"][j]].append(j)

    # templates
    templates = {}
    for w in words:
        seqs = [fp_seq["fp"][j] for j in by_word_s.get(w, [])]
        templates[w] = dba(seqs)

    # eval
    rng.shuffle(scope)
    eval_idx = scope[: min(EVAL_CAP, len(scope))]
    chance = 1.0 / len(words)
    nn_hits = nn_n = 0
    tpl_hits = tpl_n = 0
    for j_q in eval_idx:
        qs = fp_seq["speaker"][j_q]
        # NN
        gal = [(j, fp_seq["word"][j]) for j in scope if fp_seq["speaker"][j] != qs]
        if len(gal) >= 5:
            gal.sort(key=lambda gw: dtw_cost(fp_seq["fp"][j_q], fp_seq["fp"][gw[0]]))
            nn_hits += gal[0][1] == fp_seq["word"][j_q]
            nn_n += 1
        # template (leave-one-out)
        qw = fp_seq["word"][j_q]
        best_d, best_w = 2.0, None
        for w in words:
            if w == qw:
                tmpl = dba([fp_seq["fp"][j] for j in by_word_s[w] if j != j_q])
            else:
                tmpl = templates.get(w)
            if tmpl is None:
                continue
            d = dtw_cost(fp_seq["fp"][j_q], tmpl)
            if d < best_d:
                best_d, best_w = d, w
        tpl_hits += best_w == qw
        tpl_n += 1

    nn_top1 = nn_hits / max(nn_n, 1)
    tpl_top1 = tpl_hits / max(tpl_n, 1)
    delta = tpl_top1 - nn_top1
    print(f"  VTLN-fp raw cross-speaker NN top-1   : {nn_top1:.3f}  (n={nn_n})")
    print(f"  VTLN-fp DBA consensus template top-1 : {tpl_top1:.3f}  (n={tpl_n})")
    print(f"  chance (1/#words)                    : {chance:.3f}")
    print(f"  Δ(template − NN) = {delta:+.3f}")
    # compare to raw fp (no VTLN) §27: NN 0.212, template 0.172, Δ −0.040
    print("\n  §27 raw fp (no VTLN): NN 0.212, template 0.172, Δ −0.040")
    print(
        f"  VTLN normalisation {'IMPROVED' if delta > -0.040 else 'did NOT improve'} the template-vs-NN gap."
    )
    if delta > 0:
        print(
            "  → VTLN successfully removed the physical speaker entanglement → DBA averaging works."
        )
    print(
        "\n  VTLN = deterministic freq-axis warp (resample by α), physics-grounded, species-agnostic."
    )
    print("  semantic_validation stays none.")
    print("\nLS VTLN TEMPLATE DONE")


if __name__ == "__main__":
    main()
