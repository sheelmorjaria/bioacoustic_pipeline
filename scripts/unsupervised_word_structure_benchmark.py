#!/usr/bin/env python3
"""Unsupervised word-structure benchmark on LibriSpeech (HECB vs SSL discrete units).

The decisive comparison: is HECB's boundary-aware internal sub-unit sequence
competitive with modern SSL discrete units (HuBERT/wav2vec2 k-means) for
unsupervised word-structure discovery? Four unit sources run on the SAME GT word
spans (wav2vec2 forced-aligned, the GT yardstick — measurement only, not a model):

    HECB       — NBD internal sub-units per word → fingerprint clusters (k=K_FP)
    HuBERT-km  — HuBERT frame units, k-means k=K_SSL, collapse repeats
    CPC-km     — in-house CPC frame units, k-means k=K_SSL, collapse repeats
    fingerprint— uniform-window frame fingerprints → clusters (k=K_FP), collapse

Per source: within-word vs across-word normalized edit distance + ratio + nearest-
neighbour word accuracy. Controls: duration-matched across-word, phoneme-count-
matched across-word, speaker-held-out NN (does structure, not voice, drive NN?).
Plus motif→word enrichment (top bigram Cramér's V vs word identity).

This is a controlled HUMAN-SPEECH word-structure benchmark (LibriSpeech has word
GT) — "word" is honest here (unlike animal recurrent terms). It tests the
*method*; the firewall/ceiling framing is unchanged.

Run:  python scripts/unsupervised_word_structure_benchmark.py --n-files 250

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bioacoustic_pipeline.diagnostics import acoustic_fingerprint as afp  # noqa: E402

import hecb_librispeech as L  # noqa: E402

K_SSL = 50  # HuBERT/CPC k-means (zero-resource standard; k=15 failed in earlier run)
K_FP = 32  # fingerprint/HECB fingerprint-cluster vocabulary


def edit_dist_norm(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0 if m == n else 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    return dp[n] / max(m, n)


def collapse_repeats(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def kmeans_units(frames, k, seed=0):
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler().fit(frames)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=4096).fit(
        scaler.transform(frames)
    )
    sv = scaler.transform(frames)
    return np.argmin(((sv[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2), axis=1)


def main():
    ap = argparse.ArgumentParser(description="LibriSpeech unsupervised word-structure benchmark")
    ap.add_argument("--n-files", type=int, default=200)
    ap.add_argument("--k-ssl", type=int, default=K_SSL)
    ap.add_argument("--k-fp", type=int, default=K_FP)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument(
        "--max-words",
        type=int,
        default=1200,
        help="cap frequent-word instances for the O(n^2) NN metric",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading wav2vec2 aligner + HuBERT + CPC...", flush=True)
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    fa_model, labels = bundle.get_model(), bundle.get_labels()
    hubert = torchaudio.pipelines.HUBERT_BASE.get_model().to(device).eval()
    from bioacoustic_pipeline.embeddings.backbones import CPCBackbone  # noqa: PLC0415,E402

    cpc = CPCBackbone()
    cpc_ok = cpc.available
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="human").combined  # noqa: E731

    files = L.sample_files(args.n_files, args.seed)
    # word records per source: list of (word, speaker, dur_ms, n_phon, [unit_vectors or frames])
    recs = []  # generic: each = dict(word,speaker,dur,nphon, hecb_fps, fp_fps, hubert_frames, cpc_frames)
    for fi, flac in enumerate(files):
        from bioacoustic_pipeline.utils.audio import load_audio  # noqa: PLC0415

        audio, _ = load_audio(flac, target_sr=L.SR)
        if audio.size == 0:
            continue
        tr = L.load_librispeech_transcript(flac)
        if not tr:
            continue
        res = L.forced_align(tr, audio, L.SR, model=fa_model, labels=labels)
        if not res.word_spans:
            continue
        speaker = Path(flac).parent.parent.name
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            hubert_all = hubert(wav)[0].squeeze(0).cpu().numpy()  # (T, 768) @100fps
        cpc_all = cpc.embed_frames(audio, L.SR) if cpc_ok else None  # (Tc, ~256) @100fps
        for ws, we, word in res.word_spans:
            if we - ws < 80:
                continue
            fs, fe = int(ws / 1000 * 100), int(we / 1000 * 100)
            hecb_fps, win_fps = [], []
            # HECB: NBD internal proposal fingerprints
            for p in L.word_internal_proposals(audio, L.SR, ws, we):
                si = max(0, int(p.time_ms / 1000 * L.SR) - int(0.02 * L.SR))
                sj = si + int(0.04 * L.SR)
                seg = audio[si:sj]
                if len(seg) >= int(0.012 * L.SR):
                    v = np.asarray(fp_fn(seg, L.SR), dtype=np.float32)
                    if np.any(v):
                        hecb_fps.append(v)
            # fingerprint: uniform 30ms-hop windows
            hop = int(0.03 * L.SR)
            win = int(0.04 * L.SR)
            for s in range(int(ws / 1000 * L.SR), int(we / 1000 * L.SR) - win, hop):
                seg = audio[s : s + win]
                if len(seg) >= win:
                    v = np.asarray(fp_fn(seg, L.SR), dtype=np.float32)
                    if np.any(v):
                        win_fps.append(v)
            hf = hubert_all[fs:fe] if fe > fs + 1 and fe <= len(hubert_all) else None
            cf = None
            if cpc_all is not None:
                cs = min(int(ws / 1000 * 100), len(cpc_all) - 1)
                ce = max(cs + 1, min(int(we / 1000 * 100), len(cpc_all)))
                cf = cpc_all[cs:ce] if ce > cs + 1 else None
            nphon = len([p for p in res.phoneme_starts if ws + 50 < p < we - 50])
            if hecb_fps and win_fps and hf is not None and len(hf) >= 2:
                recs.append(
                    dict(
                        word=word.upper(),
                        speaker=speaker,
                        dur=we - ws,
                        nphon=nphon,
                        hecb=hecb_fps,
                        fp=win_fps,
                        hubert=hf,
                        cpc=cf,
                    )
                )
        if (fi + 1) % 50 == 0:
            print(f"  {fi + 1}/{len(files)} files → {len(recs)} words", flush=True)
    print(f"  {len(recs)} word instances", flush=True)

    # ── Build per-source cluster sequences (global k-means over each unit pool) ──
    def seqs_from(name, all_units, per_word_units, k, collapse=True):
        if len(all_units) < k * 5:
            print(f"  [{name}] too few units ({len(all_units)})")
            return [None] * len(per_word_units)
        u = kmeans_units(np.asarray(all_units, dtype=np.float32), k)
        out, idx = [], 0
        for vu in per_word_units:
            n = len(vu)
            tok = u[idx : idx + n].tolist()
            idx += n
            out.append(collapse_repeats(tok) if collapse else tok)
        return out

    hecb_all = [v for r in recs for v in r["hecb"]]
    fp_all = [v for r in recs for v in r["fp"]]
    hubert_all = [v for r in recs for v in r["hubert"]]
    cpc_allu = [v for r in recs for v in (r["cpc"] if r["cpc"] is not None else [])]
    hecb_seq = seqs_from("HECB", hecb_all, [r["hecb"] for r in recs], args.k_fp)
    fp_seq = seqs_from("fingerprint", fp_all, [r["fp"] for r in recs], args.k_fp)
    hubert_seq = seqs_from("HuBERT", hubert_all, [r["hubert"] for r in recs], args.k_ssl)
    cpc_seq = (
        seqs_from("CPC", cpc_allu, [r["cpc"] for r in recs], args.k_ssl)
        if cpc_ok
        else [None] * len(recs)
    )

    sources = [
        ("HECB", hecb_seq),
        (f"HuBERT-km({args.k_ssl})", hubert_seq),
        (f"CPC-km({args.k_ssl})", cpc_seq),
        (f"fingerprint({args.k_fp})", fp_seq),
    ]

    # ── Per-source metrics ──
    words = [r["word"] for r in recs]
    freq = {w for w, c in Counter(words).items() if c >= args.min_freq}
    keep = [i for i, r in enumerate(recs) if r["word"] in freq]
    if len(keep) > args.max_words:
        random.Random(args.seed).shuffle(keep)
        keep = sorted(keep[: args.max_words])
    print(
        f"  frequent words: {len(freq)} types; kept {len(keep)} instances (cap {args.max_words})",
        flush=True,
    )

    def within_across_nn(seqs, idxs):
        rng = random.Random(0)
        within, across = [], []
        across_dur, across_phon = [], []
        nn_correct = nn_total = 0
        nn_xspk_correct = nn_xspk_total = 0
        for i in idxs:
            wi = seqs[i]
            if not wi:
                continue
            best_d, best_w, best_spk = 2.0, "?", True
            for j in idxs:
                if i == j or not seqs[j]:
                    continue
                d = edit_dist_norm(wi, seqs[j])
                if d < best_d:
                    best_d, best_w = d, recs[j]["word"]
                    best_spk = recs[j]["speaker"] == recs[i]["speaker"]
            if best_w != "?":
                nn_total += 1
                nn_correct += best_w == recs[i]["word"]
                if not best_spk:  # NN is a different speaker → structure, not voice
                    nn_xspk_total += 1
                    nn_xspk_correct += best_w == recs[i]["word"]
            w = recs[i]["word"]
            for j in idxs:
                if j <= i or recs[j]["word"] != w or not seqs[j]:
                    continue
                within.append(edit_dist_norm(wi, seqs[j]))
        # random across + matched across
        shuf = list(idxs)
        for _ in range(len(idxs)):
            a, b = rng.choice(shuf), rng.choice(shuf)
            if a == b or not seqs[a] or not seqs[b]:
                continue
            d = edit_dist_norm(seqs[a], seqs[b])
            across.append(d)
            if abs(recs[a]["dur"] - recs[b]["dur"]) < 0.25 * max(recs[a]["dur"], recs[b]["dur"]):
                across_dur.append(d)
            if abs(recs[a]["nphon"] - recs[b]["nphon"]) <= 1:
                across_phon.append(d)
        return (
            float(np.mean(within)) if within else 0,
            float(np.mean(across)) if across else 0,
            float(np.mean(across_dur)) if across_dur else 0,
            float(np.mean(across_phon)) if across_phon else 0,
            nn_correct / max(nn_total, 1),
            nn_xspk_correct / max(nn_xspk_total, 1),
            len(within),
        )

    def motif_enrich(seqs, idxs):
        """Top-bigram Cramér's V of bigram-presence vs word identity (motif→word)."""
        from scipy.stats import chi2_contingency  # noqa: PLC0415

        bg = Counter()
        for i in idxs:
            s = seqs[i]
            if not s:
                continue
            for k in range(len(s) - 1):
                bg[(s[k], s[k + 1])] += 1
        top = [m for m, _ in bg.most_common(20)]
        words = [recs[i]["word"] for i in idxs]
        uw = sorted(set(words))
        if len(uw) < 2:
            return 0.0, 0.0
        Vs = []
        for motif in top:
            has = [
                any(
                    seqs[i][k] == motif[0] and seqs[i][k + 1] == motif[1]
                    for k in range(len(seqs[i]) - 1)
                )
                if seqs[i]
                else False
                for i in idxs
            ]
            table = np.zeros((2, len(uw)))
            for h, w in zip(has, words):
                table[int(h), uw.index(w)] += 1
            if table.sum() < 10 or table.sum(axis=0).min() == 0:
                continue
            try:
                chi2, _, _, _ = chi2_contingency(table)
                V = float(np.sqrt(chi2 / (table.sum() * (min(2, len(uw)) - 1))))
                Vs.append(min(V, 1.0))
            except Exception:  # noqa: BLE001
                pass
        return (
            float(np.mean(Vs)) if Vs else 0.0,
            float(np.mean([v > 0.1 for v in Vs])) if Vs else 0.0,
        )

    print(f"\n{'=' * 92}")
    print(
        f"UNSUPERVISED WORD-STRUCTURE BENCHMARK — LibriSpeech (n={len(keep)} words of ≥{args.min_freq} freq)"
    )
    print(f"{'=' * 92}")
    print(
        f"  {'source':18} {'within':>7} {'across':>7} {'a_dur':>6} {'a_phn':>6} {'ratio':>6} "
        f"{'NN':>6} {'NN_xsp':>7} {'motifV':>7}"
    )
    print("  " + "-" * 86)
    for name, seqs in sources:
        if any(s is None for s in seqs):
            print(f"  {name:18} (insufficient data)")
            continue
        w, a, ad, ap, nn, nnx, npairs = within_across_nn(seqs, keep)
        mV, _ = motif_enrich(seqs, keep)
        print(
            f"  {name:18} {w:7.3f} {a:7.3f} {ad:6.3f} {ap:6.3f} {w / max(a, 1e-6):6.3f} "
            f"{nn:6.3f} {nnx:7.3f} {mV:7.3f}"
        )

    print("\n  Controls: a_dur/a_phn = across-word baseline matched on duration / phoneme-count")
    print(
        "            NN_xsp = NN accuracy restricting to cross-speaker neighbours (structure, not voice)"
    )
    print(
        "            motifV = mean Cramér's V of top-20 bigrams vs word identity (motif→word enrichment)"
    )
    print("\nWORD-STRUCTURE BENCHMARK DONE")


if __name__ == "__main__":
    main()
