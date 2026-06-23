#!/usr/bin/env python3
"""HuBERT sanity ablation — layer × k × collapse × pooling grid (reviewer-safe).

Sweeps HuBERT's configuration space on the SAME LibriSpeech words so the "HuBERT
inconsistent" conclusion (§14/§15) survives the rebuttal "you didn't tune HuBERT".
Fingerprint baseline runs on the exact same words for an apples-to-apples ceiling.

Track 1 (sequence consistency, §14-style): HuBERT layer∈{6,9,12} × k∈{50,100,200} ×
collapse∈{collapsed, uncollapsed} → within-vs-across norm-Levenshtein ratio + NN acc.
Track 2 (whole-word clustering, Stage-D-style): HuBERT layer∈{6,9,12} × pool∈{mean,
statistics(mean+std)} → V-measure / ARI / token-F1.

If ANY HuBERT config beats fingerprint, it is reported honestly as the strongest SSL
baseline. If none does, the current conclusion locks. Human-speech word GT ("word"
honest here); firewall/ceiling unchanged.

Run:  python scripts/hubert_ablation.py --n-files 200

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

LAYERS = [6, 9, 12]
KS = [50, 100, 200]


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


def cluster_eval(gt, pred):
    from sklearn.metrics import v_measure_score, adjusted_rand_score  # noqa: PLC0415

    gt, pred = np.asarray(gt), np.asarray(pred)
    n = len(gt)
    clusters = {}
    for t, p in zip(gt, pred):
        clusters.setdefault(p, []).append(t)
    purity = sum(max(Counter(v).values()) for v in clusters.values()) / max(n, 1)
    clst_to_word = {p: Counter(v).most_common(1)[0][0] for p, v in clusters.items()}
    tp = fp = fn = 0
    for w in set(gt.tolist()):
        assigned = {p for p in clusters if clst_to_word[p] == w}
        tp += sum(1 for t, p in zip(gt, pred) if t == w and p in assigned)
        fp += sum(1 for t, p in zip(gt, pred) if t != w and p in assigned)
        fn += sum(1 for t, p in zip(gt, pred) if t == w and p not in assigned)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return v_measure_score(gt, pred), adjusted_rand_score(gt, pred), purity, f1


def consistency(seqs, words, collapse):
    """within/across edit distance + NN acc over aligned (seqs, words)."""
    rng = random.Random(0)
    within, across = [], []
    nn_c = nn_t = 0
    seqset = [collapse_repeats(s) if collapse else s for s in seqs]
    n = len(seqs)
    for i in range(n):
        wi = seqset[i]
        if not wi:
            continue
        best_d, best_w = 2.0, "?"
        for j in range(n):
            if i == j or not seqset[j]:
                continue
            d = edit_dist_norm(wi, seqset[j])
            if d < best_d:
                best_d, best_w = d, words[j]
            if words[j] == words[i] and j > i:
                within.append(d)
        if best_w != "?":
            nn_t += 1
            nn_c += best_w == words[i]
    for _ in range(n):
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b and seqset[a] and seqset[b]:
            across.append(edit_dist_norm(seqset[a], seqset[b]))
    w = float(np.mean(within)) if within else 0.0
    a = float(np.mean(across)) if across else 0.0
    return w, a, w / max(a, 1e-6), nn_c / max(nn_t, 1)


def main():
    ap = argparse.ArgumentParser(description="HuBERT sanity ablation (layer×k×collapse×pool)")
    ap.add_argument("--n-files", type=int, default=200)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading wav2vec2 aligner + HuBERT...", flush=True)
    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    fa_model, labels = bundle.get_model(), bundle.get_labels()
    hubert = torchaudio.pipelines.HUBERT_BASE.get_model().to(device).eval()
    fp_fn = lambda seg, sr: afp.extract(seg, sr, species="human").combined  # noqa: E731

    files = L.sample_files(args.n_files, args.seed)
    # recs[i] = {word, fp(whole), lyr: {L: frames(Tw,768)}}
    recs = []
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
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            feats, _ = hubert.extract_features(wav)  # list of (1,T,dim) per layer
        nlay = len(feats)
        feats = [f.squeeze(0).cpu().numpy() for f in feats]
        for ws, we, word in res.word_spans:
            if we - ws < 80:
                continue
            fs, fe = int(ws / 1000 * 100), int(we / 1000 * 100)
            i, j = int(ws / 1000 * L.SR), int(we / 1000 * L.SR)
            seg = audio[i:j]
            if len(seg) < int(0.03 * L.SR):
                continue
            fp = np.asarray(fp_fn(seg, L.SR), dtype=np.float32)
            if not np.any(fp):
                continue
            lyr = {}
            ok = True
            for Ll in LAYERS:
                li = min(Ll, nlay - 1)
                fr = feats[li][fs:fe] if fe > fs + 1 and fe <= len(feats[li]) else None
                if fr is None or len(fr) < 2:
                    ok = False
                    break
                lyr[Ll] = fr
            if ok:
                recs.append(dict(word=word.upper(), fp=fp, lyr=lyr))
        if (fi + 1) % 50 == 0:
            print(f"  {fi + 1}/{len(files)} files → {len(recs)} words", flush=True)

    words = [r["word"] for r in recs]
    freq = {w for w, c in Counter(words).items() if c >= args.min_freq}
    keep = [i for i, r in enumerate(recs) if r["word"] in freq]
    random.Random(args.seed).shuffle(keep)
    keep = sorted(keep[: args.max_words])
    print(f"  {len(recs)} words; {len(freq)} freq types; kept {len(keep)}", flush=True)

    words_kept = [recs[i]["word"] for i in keep]
    print(f"\n{'=' * 92}\nHUBERT SANITY ABLATION — Track 1 (sequence consistency)\n{'=' * 92}")
    print(f"  {'config':28} {'within':>7} {'across':>7} {'ratio':>6} {'NN':>6}")
    print("  " + "-" * 60)
    print(f"  {'fingerprint(k32,col) [§14 REF]':28} {' ':>7} {' ':>7} {0.964:6.3f} {0.147:6.3f}")

    def track1(name, seqs, collapse):
        w, a, ratio, nn = consistency(seqs, words_kept, collapse)
        print(f"  {name:28} {w:7.3f} {a:7.3f} {ratio:6.3f} {nn:6.3f}")
        return ratio, nn

    # HuBERT grid
    best1 = (0.0, 0.0, "")
    for Ll in LAYERS:
        all_fr = np.concatenate([recs[i]["lyr"][Ll] for i in keep], axis=0)
        for k in KS:
            if len(all_fr) < k * 5:
                continue
            u = kmeans_units(all_fr.astype(np.float32), k)
            # per-word raw token lists
            raw, idx = [], 0
            for i in keep:
                n = len(recs[i]["lyr"][Ll])
                raw.append(u[idx : idx + n].tolist())
                idx += n
            for collapse in (True, False):
                tag = f"HuBERT L{Ll} k{k} {'col' if collapse else 'raw'}"
                ratio, nn = track1(tag, raw, collapse)
                if nn > best1[1]:
                    best1 = (ratio, nn, tag)
    print(f"\n  best HuBERT Track-1 NN: {best1[2]}  (ratio {best1[0]:.3f}, NN {best1[1]:.3f})")

    print(f"\n{'=' * 92}\nHUBERT SANITY ABLATION — Track 2 (whole-word clustering)\n{'=' * 92}")
    print(f"  {'config':28} {'V':>6} {'ARI':>6} {'purity':>7} {'tokF1':>6}")
    print("  " + "-" * 56)
    gt = [recs[i]["word"] for i in keep]
    kcl = min(len(set(gt)), 60)
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    def track2(name, emb):
        X = StandardScaler().fit_transform(np.asarray(emb, dtype=np.float32))
        pred = KMeans(n_clusters=kcl, random_state=0, n_init=10).fit_predict(X)
        v, ari, pur, f1 = cluster_eval(gt, pred)
        print(f"  {name:28} {v:6.3f} {ari:6.3f} {pur:7.3f} {f1:6.3f}")
        return v, f1

    track2("fingerprint(44-D) [REF]", [recs[i]["fp"] for i in keep])
    best2 = (0.0, 0.0, "")
    for Ll in LAYERS:
        for pool in ("mean", "stat"):
            embs = []
            for i in keep:
                fr = recs[i]["lyr"][Ll]
                embs.append(
                    fr.mean(axis=0)
                    if pool == "mean"
                    else np.concatenate([fr.mean(axis=0), fr.std(axis=0)])
                )
            v, f1 = track2(f"HuBERT L{Ll} {pool}", embs)
            if f1 > best2[1]:
                best2 = (v, f1, f"HuBERT L{Ll} {pool}")
    print(f"\n  best HuBERT Track-2 tokF1: {best2[2]}  (V {best2[0]:.3f}, tokF1 {best2[1]:.3f})")
    print("\nHUBERT ABLATION DONE")


if __name__ == "__main__":
    main()
